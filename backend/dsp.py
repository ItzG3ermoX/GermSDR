from __future__ import annotations

from fractions import Fraction
from functools import lru_cache

import numpy as np
from scipy.signal import firwin, lfilter, lfilter_zi, resample_poly
from math import gcd

# The base capture rate that perceived zoom is measured against. Kept in sync
# with state.BASE_RATE — duplicated here to avoid a circular-ish import web.
_BASE_RATE = 2_400_000

try:
    from scipy.signal.windows import blackmanharris
except ImportError:
    blackmanharris = None

DEFAULT_SAMPLE_RATE = 2_400_000
DEFAULT_AUDIO_RATE  = 48_000


class DemodState:
    """Container for all DSP state that was previously module-level globals.

    Each pipeline owns one ``DemodState`` instance so that multiple SDR instances
    can coexist, tests are isolated, and state transitions (zoom/mode/rate changes)
    are explicit and safe.
    """

    __slots__ = (
        # Waterfall zoom accumulator (hi-res zoom buffer)
        "_zoom_accum", "_zoom_accum_key", "_zoom_last_bins",
        # FM discriminator continuity
        "_fm_last",
        # Per-filter lfilter state (keyed by taps id)
        "_filter_state",
        # Channelizer (IF down-conversion) FIFO states
        "_channel_state", "_channel_phase",
        # Pre-demodulation complex lowpass state
        "_pre_demod_state",
        # AGC envelope & gain
        "_agc_level", "_agc_gain",
        # FM de-emphasis one-pole state
        "_deemph_prev",
        # DC blocker IIR state
        "_dcblock_zi",
        # FM output leveller
        "_fm_level",
        # Frequency-shift phase accumulator
        "_tune_phase", "_tune_offset_active",
        # Mute ramp counter (samples remaining)
        "_mute_ramp",
        # Squelch state
        "_squelch_open",
        # Latest measured signal powers (dBFS)
        "_last_signal_power_db", "_last_if_power_db",
        # Rate/offset change detection
        "_last_demod_rate", "_last_tune_offset",
    )

    def __init__(self) -> None:
        self.reset_all()

    def reset_all(self) -> None:
        """Reset ALL DSP state. Call when capture sample rate changes."""
        # Waterfall zoom accumulator — key is (hw_rate, up, down, slice_center)
        self._zoom_accum: np.ndarray = np.empty(0, dtype=np.complex64)
        self._zoom_accum_key: tuple[int, ...] | None = None
        self._zoom_last_bins: np.ndarray | None = None
        # FM discriminator
        self._fm_last: complex = 0j
        # Filter histories
        self._filter_state: dict[int, np.ndarray] = {}
        self._channel_state: dict[tuple[int, int, int], np.ndarray] = {}
        self._channel_phase: dict[tuple[int, int, int], int] = {}
        self._pre_demod_state: dict[tuple[int, ...], np.ndarray] = {}
        # AGC
        self._agc_level: float = 1e-3
        self._agc_gain: float = 1.0
        # De-emphasis
        self._deemph_prev: float = 0.0
        # DC blocker
        self._dcblock_zi: np.ndarray | None = None
        # FM leveller
        self._fm_level: float = 0.25
        # Tuning phase
        self._tune_phase: float = 0.0
        self._tune_offset_active: float = 0.0
        # Mute ramp
        self._mute_ramp: int = 0
        # Squelch
        self._squelch_open: bool = True
        # Signal power
        self._last_signal_power_db: float = -160.0
        self._last_if_power_db: float = -160.0
        # Rate/offset tracking
        self._last_demod_rate: int | None = None
        self._last_tune_offset: float = 0.0

    # --- Convenience property accessors for module-level API compat ---

    @property
    def squelch_open(self) -> bool:
        return self._squelch_open

    @property
    def last_signal_power_db(self) -> float:
        return self._last_signal_power_db

    @property
    def last_if_power_db(self) -> float:
        return self._last_if_power_db


# Default module-level state used when no DemodState is passed explicitly.
# Functions that accept an optional ``state: DemodState`` parameter will use
# this instance as a fallback, preserving the old module-global API while
# allowing callers (pipeline.py) to pass their own.
_default_state = DemodState()


@lru_cache(maxsize=16)
def _window(fft_size: int, name: str = "blackmanharris") -> np.ndarray:
    if name == "blackmanharris" and blackmanharris is not None:
        return blackmanharris(fft_size, sym=False).astype(np.float32)
    return np.hanning(fft_size).astype(np.float32)


def compute_waterfall(
    iq: np.ndarray,
    fft_size: int = 32_768,
    *,
    window: str = "blackmanharris",
) -> np.ndarray:
    """Return fft_size float32 bins in dBFS, DC-centred."""
    samples = np.asarray(iq, dtype=np.complex64)
    if samples.size and samples.size < fft_size:
        # Zoomed slice: fewer real samples than fft_size. Window only the
        # populated region and normalise by that window's coherent gain
        # (sum of taps), NOT by fft_size. Dividing by fft_size here scaled the
        # spectrum down by ~10*log10(n/fft_size) dB -- which is exactly why the
        # waterfall went much darker the more you zoomed in. Windowing the real
        # region keeps the dB level consistent with the full (1x) view.
        win = _window(samples.size, window)
        norm = float(np.sum(win))
        seg = np.zeros(fft_size, dtype=np.complex64)
        seg[: samples.size] = samples * win
        spectrum = np.fft.fftshift(np.fft.fft(seg, fft_size))
    else:
        segment = samples[:fft_size] if samples.size else np.zeros(fft_size, dtype=np.complex64)
        win = _window(fft_size, window)
        norm = float(np.sum(win))
        spectrum = np.fft.fftshift(np.fft.fft(segment * win, fft_size))

    power_db = 20.0 * np.log10(np.abs(spectrum) / max(norm, 1e-9) + 1e-12)
    return power_db.astype(np.float32)


def compute_waterfall_zoom(
    iq: np.ndarray,
    fft_size: int,
    sample_rate: int,
    center_freq: float,
    *,
    zoom: float = 1.0,
    pan: float = 0.5,
    window: str = "blackmanharris",
    state: DemodState = _default_state,
) -> tuple[np.ndarray, float, float]:
    """
    Server-side zoom: render only the visible slice, at full FFT resolution.

    The captured band is center_freq +/- sample_rate/2. ``pan`` (0..1) is the
    centre of the visible window across that band and ``zoom`` shrinks its width.
    When zoom > 1 we digitally down-convert that slice to baseband, low-pass +
    decimate it, then FFT ``fft_size`` bins over the narrowed band -- so the same
    number of bins now covers a much smaller span (finer Hz/bin = sharper).

    Returns (bins_dBFS, slice_center_freq_hz, slice_sample_rate_hz). The caller
    puts the returned centre/rate in the frame header so all downstream
    frequency math (ruler, click-to-tune) keeps working unchanged.
    """
    samples = np.asarray(iq, dtype=np.complex64)
    if zoom <= 1.0 or samples.size == 0:
        return compute_waterfall(samples, fft_size, window=window), float(center_freq), float(sample_rate)

    zoom = float(min(zoom, 64.0))

    # Visible-window centre as a normalised offset from DC, in (-0.5, 0.5).
    offset = (float(pan) - 0.5)
    # Frequency (Hz) of the slice centre, relative to the captured centre.
    freq_shift = offset * sample_rate
    slice_center = float(center_freq) + freq_shift

    # Mix the slice down to baseband: multiply by exp(-j*2*pi*freq_shift/fs*n).
    n = np.arange(samples.size, dtype=np.float64)
    mixer = np.exp(-2j * np.pi * (freq_shift / sample_rate) * n).astype(np.complex64)
    shifted = (samples * mixer).astype(np.complex64)

    # FRACTIONAL decimation so the slice width is EXACTLY sample_rate/zoom -- this
    # is what makes the on-screen frequency scale match the zoom the user picked.
    # A rounded integer decim (the old approach) made the displayed span disagree
    # with the rendered width, so the waterfall looked stretched/soft at any
    # non-integer zoom (and the slider produces fractional zooms constantly).
    # resample_poly(up, down) resamples by up/down; choosing up/down ~= 1/zoom
    # with a small denominator keeps the polyphase filter cheap and steep.
    up, down = _zoom_ratio(zoom)
    narrowed = resample_poly(shifted, up, down).astype(np.complex64)
    slice_rate = sample_rate * up / down

    # --- High-resolution zoom via a sliding accumulator -------------------
    # One decimated block only yields ~block/zoom samples, so the FFT shrinks as
    # fast as the span does -> Hz/bin stays constant and the deep-zoom view never
    # actually gets finer (just a smaller slice of the same coarse grid, upsampled
    # to the display = soft). To get genuine high resolution we KEEP a sliding
    # buffer of the most-recent decimated samples and FFT a large fixed window of
    # it, so Hz/bin keeps shrinking with zoom. The buffer slides forward every
    # frame (always includes the freshest block -> no staleness/lag) and resets
    # the instant zoom or pan changes (key change) so bands never mix.
    #
    # We use the PERCEIVED zoom (BASE_RATE / slice_rate) for the target FFT size
    # so Hz/bin is continuous across hardware rate boundaries. The digital zoom
    # (the ``zoom`` parameter) resets at each boundary, which would halve the
    # target FFT and make the waterfall look less sharp after a rate switch even
    # though the visible span hasn't changed.
    perceived_zoom = float(_BASE_RATE) / float(slice_rate) if slice_rate > 0 else zoom
    target_nfft = _zoom_target_nfft(perceived_zoom, fft_size)
    # Key includes the hardware rate so a hw-rate auto-switch (and the new band
    # it implies) always flushes the accumulator -- never mix samples from two
    # different captured bands.
    accum_key = (int(sample_rate), up, down, int(round(slice_center)))
    bins = _accumulate_and_transform(narrowed, target_nfft, fft_size, accum_key, window, state=state)
    return bins, slice_center, float(slice_rate)


def _zoom_ratio(zoom: float) -> tuple[int, int]:
    """Rational up/down for resample_poly giving a decimation as close to ``zoom``
    as possible with fine granularity. We pick ``up`` dynamically based on zoom
    so each unit of zoom changes the ratio, avoiding the old "sticky" feel where
    the ratio only jumped every ~0.5 zoom units. Ratio granularity improves as
    zoom increases (up to 1/16 zoom unit steps)."""
    ratio = Fraction(1.0 / max(1.0, zoom)).limit_denominator(256)
    return ratio.numerator, ratio.denominator


def _zoom_target_nfft(zoom: float, fft_size: int) -> int:
    """FFT length for the zoomed slice. To keep getting FINER Hz/bin as you zoom
    (instead of just a smaller slice of the same coarse grid), the target grows
    with zoom and is allowed to exceed the display width -- we then peak-reduce
    the extra resolution onto the display, so narrow signals stay crisp. Capped
    for cost.

    Resolution intuition: span = sample_rate/zoom, so Hz/bin = sample_rate /
    (zoom * nfft). Holding nfft constant keeps Hz/bin constant (no gain from
    zooming). Letting nfft grow ~linearly with zoom is what actually sharpens
    the deep-zoom view."""
    # Grow ~linearly with zoom (finer Hz/bin), but never below the display width.
    want = max(float(fft_size), 2048.0 * zoom)
    p2 = 1 << int(np.ceil(np.log2(max(256.0, want))))
    return int(min(p2, 131072))


# Per-decim accumulation buffer of narrowed (decimated) IQ, so a large FFT can
# be built up across several capture blocks at high zoom. Reset whenever the
# decimation factor changes (zoom change) so stale samples never mix in.
# (Now stored in DemodState._zoom_accum / _zoom_accum_key / _zoom_last_bins.)


def _accumulate_and_transform(
    narrowed: np.ndarray,
    target_nfft: int,
    out_size: int,
    key: tuple[int, ...],
    window: str,
    state: DemodState = _default_state,
) -> np.ndarray:
    if key != state._zoom_accum_key:
        state._zoom_accum = np.empty(0, dtype=np.complex64)
        state._zoom_accum_key = key
        state._zoom_last_bins = None

    # Append the fresh decimated block and keep only the most recent
    # target_nfft samples (a sliding window). Bound to target_nfft so memory and
    # FFT cost stay fixed and the window always ends on the freshest data.
    state._zoom_accum = np.concatenate((state._zoom_accum, narrowed))
    if state._zoom_accum.size > target_nfft:
        state._zoom_accum = state._zoom_accum[-target_nfft:].copy()

    if state._zoom_accum.size < target_nfft:
        # Still filling: FFT what we have so far (at its natural size) so the
        # waterfall keeps scrolling and progressively sharpens as it fills,
        # instead of stalling or showing a stale frame.
        bins = _hi_res_bins(state._zoom_accum, out_size, window=window)
        state._zoom_last_bins = bins
        return bins

    # Full window: one large FFT every frame over the freshest target_nfft
    # samples -> finest Hz/bin for this zoom, refreshed continuously (no lag).
    bins = _hi_res_bins(state._zoom_accum, out_size, window=window, force_nfft=target_nfft)
    state._zoom_last_bins = bins
    return bins


def _hi_res_bins(
    iq: np.ndarray,
    out_size: int,
    *,
    window: str = "blackmanharris",
    force_nfft: int | None = None,
) -> np.ndarray:
    """Full-resolution dBFS spectrum of a (decimated) slice, resampled to
    ``out_size`` display bins.

    Runs a true FFT (resolution = rate / nfft), then maps the bins onto the
    display width with peak-preserving reduction when downscaling. ``force_nfft``
    pins the transform length (used by the accumulator for a fixed large FFT).
    """
    samples = np.asarray(iq, dtype=np.complex64)
    n = samples.size
    if n == 0:
        return np.full(out_size, -160.0, dtype=np.float32)

    if force_nfft is not None:
        nfft = int(force_nfft)
        if n < nfft:
            seg = np.zeros(nfft, dtype=np.complex64)
            seg[:n] = samples
        else:
            seg = samples[:nfft]
    else:
        # Largest power-of-two FFT that fits the data -> clean full-resolution
        # transform without zero-pad interpolation artefacts.
        nfft = 1 << int(np.floor(np.log2(n)))
        nfft = max(256, nfft)
        seg = samples[:nfft]

    win = _window(nfft, window)
    norm = float(np.sum(win))
    spectrum = np.fft.fftshift(np.fft.fft(seg * win, nfft))
    power_db = (20.0 * np.log10(np.abs(spectrum) / max(norm, 1e-9) + 1e-12)).astype(np.float32)

    if nfft == out_size:
        return power_db
    # Downscale: peak-preserving reduction using vectorised group-reduce.
    # Build a group index of size nfft so group[j] = the output bin index
    # that input bin j maps to. Then np.maximum.at reduces each group to
    # a single max value in a single C-level pass (no Python loop).
    idx = np.linspace(0, nfft, out_size + 1).astype(np.int64)
    idx[-1] = min(idx[-1], nfft)
    group = np.zeros(nfft, dtype=np.int32)
    group[idx[1:-1]] = 1
    group = np.cumsum(group, dtype=np.int32)
    out = np.full(out_size, -160.0, dtype=np.float32)
    np.maximum.at(out, group, power_db)
    return out


@lru_cache(maxsize=64)
def _decim_lowpass(decim_stage: int, taps: int) -> np.ndarray:
    """Steep anti-alias FIR for one decimation stage, normalised to the stage's
    input Nyquist. Cutoff sits at 0.8/decim_stage so the passband is preserved
    while the stopband lands inside the next Nyquist. Blackman window => deep
    stopband (low aliasing) and the tap count gives a tight transition."""
    cutoff = min(0.95, 0.85 / decim_stage)
    return firwin(taps, cutoff, window="blackman").astype(np.float32)


def _decimate_complex_stateful(iq: np.ndarray, decim: int, state: DemodState = _default_state) -> np.ndarray:
    """Like ``_decimate_complex`` but carries each stage's FIR state across
    blocks (keyed by stage ratio + tap count) so the IF channelizer is
    phase-continuous from one capture block to the next -- no per-block click.

    Stage order matters for CPU cost, not just filter count. Each stage's FIR
    has ``64 * step`` taps and runs over WHATEVER data size is current at that
    point. Picking the LARGEST available factor first (e.g. a single 8x stage
    for decim=8) runs the biggest filter (513 taps) over the FULL-RATE block --
    the most expensive combination possible. At 2.4 Msps/65536-sample blocks
    this ate ~56% of the real-time budget on modest hardware, and any extra
    load (waterfall FFT thread, other clients) pushed it over 100%, producing
    dropped/late audio blocks that sound like glitchy, wrong-sample-rate audio
    -- exactly the "1x-3x sounds weird, 3x+ sounds fine" symptom (deeper zoom
    switches to a lower hardware rate with a cheaper decim factor, masking the
    bug). Preferring the SMALLEST factor first instead shrinks the array after
    the very first (cheapest, small-tap) stage, so every subsequent stage's
    bigger filter runs over far fewer samples -- standard multistage
    decimation practice, and it cuts total multiply-adds roughly in half for
    decim=8 (three cheap 2x stages instead of one big 8x stage).
    """
    work = np.asarray(iq, dtype=np.complex64)
    remaining = int(decim)
    stage_index = 0
    while remaining > 1:
        step = next((f for f in (2, 3, 4, 5, 6, 7, 8) if remaining % f == 0), remaining)
        taps = min(513, 64 * step) | 1
        h = _decim_lowpass(step, taps)
        key = (stage_index, step, taps)
        zi = state._channel_state.get(key)
        if zi is None or zi.size != taps - 1:
            zi = np.zeros(taps - 1, dtype=np.complex64)
        filtered, zf = lfilter(h, [1.0], work, zi=zi)
        state._channel_state[key] = zf.astype(np.complex64)
        phase = state._channel_phase.get(key, 0)
        start = (-phase) % step
        input_size = work.size
        work = filtered.astype(np.complex64)[start::step]
        remaining //= step
        state._channel_phase[key] = (phase + input_size) % step
        stage_index += 1
    return work.astype(np.complex64)


def _channelize(
    iq: np.ndarray, sample_rate: int, if_rate: int, state: DemodState = _default_state
) -> tuple[np.ndarray, int]:
    """Filter + integer-decimate baseband IQ down toward ``if_rate`` before a
    non-linear detector runs.

    The tuned signal has already been mixed to DC by ``_frequency_shift``. At the
    full capture rate (e.g. 2.4 MHz) the FM discriminator / AM envelope detector
    sees the WHOLE band, so adjacent stations and wideband noise fold into the
    audio (this is the "weird audio until you zoom" + "clicking a few kHz over
    still hears the old station" bug). Decimating the IQ to a narrow IF first
    isolates the wanted channel so the detector only sees it.

    Returns (iq_if, if_rate_actual). When the capture rate is already at/below the
    target IF (deep zoom, where the hardware rate ladder already narrowed the
    band) we pass through unchanged -- which is exactly why audio already worked
    at high zoom.
    """
    samples = np.asarray(iq, dtype=np.complex64)
    if sample_rate <= if_rate or samples.size == 0:
        return samples, int(sample_rate)
    decim = max(1, int(sample_rate // if_rate))
    if decim <= 1:
        return samples, int(sample_rate)
    narrowed = _decimate_complex_stateful(samples, decim, state=state)
    return narrowed, int(sample_rate // decim)





def _filter_complex_stateful(taps: np.ndarray, data: np.ndarray, state: DemodState = _default_state) -> np.ndarray:
    """Apply a real FIR filter to complex IQ with state continuity across blocks.

    The same real FIR is applied to I and Q independently by lfilter (scipy
    handles complex input with real coefficients natively). State is keyed by
    (taps_id,) and carried across blocks so there is no click at boundaries.
    """
    key = (id(taps),)
    zi = state._pre_demod_state.get(key)
    if zi is None or zi.size != taps.size - 1:
        zi = np.zeros(taps.size - 1, dtype=np.complex64)
    out, zf = lfilter(taps, [1.0], data, zi=zi)
    state._pre_demod_state[key] = zf.astype(np.complex64)
    return out.astype(np.complex64)


def _reset_audio_dsp_state(state: DemodState = _default_state) -> None:
    """Clear all IIR/FIR/phase/AGC state when the capture sample rate changes.

    This is critical: when the user zooms (hardware rate ladder changes sample_rate),
    the old filter states, FM discriminator history, de-emphasis, DC blocker and
    phase accumulators were computed for a completely different sample rate.
    Keeping them produces garbage audio (crackles, wrong pitch, pumping, etc.)
    until the states slowly decay. Resetting gives a clean start at the new rate.
    """
    state.reset_all()
    # Note: _last_demod_rate is updated by the caller after reset.


def _fm_demod(iq: np.ndarray, state: DemodState = _default_state) -> np.ndarray:
    """Continuous polar-discriminator FM demod.

    Prepends the previous block's final sample so the differential at the block
    boundary is correct and the output length equals the input length.
    """
    if iq.size == 0:
        return np.empty(0, dtype=np.float32)
    prev = np.empty(iq.size + 1, dtype=np.complex64)
    prev[0] = state._fm_last
    prev[1:] = iq
    out = np.angle(prev[1:] * np.conj(prev[:-1])).astype(np.float32)
    state._fm_last = complex(iq[-1])
    return out


# Per-filter lfilter state, keyed by the filter's taps id, carried across blocks
# so the lowpass doesn't restart cold each block (another crackle source).


def _lfilter_stateful(taps: np.ndarray, data: np.ndarray, state: DemodState = _default_state) -> np.ndarray:
    key = id(taps)
    zi = state._filter_state.get(key)
    if zi is None:
        zi = lfilter_zi(taps, [1.0]).astype(np.float32) * (data[0] if data.size else 0.0)
    out, zf = lfilter(taps, [1.0], data, zi=zi)
    state._filter_state[key] = zf
    return out.astype(np.float32)


@lru_cache(maxsize=32)
def _lowpass(sample_rate: int, cutoff_hz: int, taps: int = 127) -> np.ndarray:
    nyquist = sample_rate / 2.0
    normalized = min(0.99, cutoff_hz / nyquist)
    return firwin(taps, normalized, window="hamming").astype(np.float32)


@lru_cache(maxsize=8)
def _resample_factors(sample_rate: int, audio_rate: int):
    """Return (up, down) coprime factors for resample_poly."""
    g = gcd(audio_rate, sample_rate)
    return audio_rate // g, sample_rate // g


def _agc(audio: np.ndarray, target: float = 0.3, smoothing: float = 0.2, state: DemodState = _default_state) -> np.ndarray:
    """
    Vectorised AGC with no block-boundary step.

    A single gain-per-block makes the gain jump at every block edge (~37 Hz with
    65 k blocks), which is audible as a buzz. Instead we compute this block's
    target gain, then *ramp* the gain linearly from the previous block's gain to
    this one across the samples. That removes both the per-sample Python loop
    (the old lag) and the stair-step buzz.
    """
    if audio.size == 0:
        return audio.astype(np.float32)

    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)))) + 1e-9
    # Exponential smoothing of the envelope across blocks.
    state._agc_level = (1.0 - smoothing) * state._agc_level + smoothing * rms
    target_gain = target / max(state._agc_level, 1e-6)

    # Linear gain ramp from last block's gain -> this block's gain.
    ramp = np.linspace(state._agc_gain, target_gain, audio.size, dtype=np.float32)
    state._agc_gain = target_gain
    return np.clip(audio * ramp, -1.0, 1.0).astype(np.float32)


def _decimate(audio: np.ndarray, sample_rate: int, audio_rate: int) -> np.ndarray:
    """
    Resample to the audio rate using staged polyphase decimation.

    A single resample_poly(up=1, down=50) at 2.4 MHz is extremely slow because
    the polyphase filter is huge. Decimating in stages keeps each filter small:
    factor the integer decimation into smaller pieces, then a final fractional
    resample_poly handles any non-integer remainder.
    """
    if sample_rate == audio_rate:
        return np.asarray(audio, dtype=np.float32)

    work = np.asarray(audio, dtype=np.float32)
    rate = sample_rate

    # Stage down by integer factors (<= 10 each) while the rate stays a clean
    # multiple of the audio rate. This is the bulk of a 2.4M -> 48k decimation.
    while rate % audio_rate == 0 and rate > audio_rate:
        factor = rate // audio_rate
        step = next((f for f in (10, 8, 6, 5, 4, 3, 2) if factor % f == 0), factor)
        if step == 1:
            break
        work = resample_poly(work, 1, step).astype(np.float32)
        rate //= step

    # Final fractional correction (e.g. when sample_rate isn't a clean multiple).
    if rate != audio_rate:
        up, down = _resample_factors(rate, audio_rate)
        work = resample_poly(work, up, down).astype(np.float32)

    return work.astype(np.float32)


# (state is carried in DemodState._deemph_prev / _dcblock_zi)

def _dc_block(x: np.ndarray, r: float = 0.9995, state: DemodState = _default_state) -> np.ndarray:
    """Remove DC / very-low-freq offset with the standard DC-blocker IIR
    (b = [1, -1], a = [1, -r]). The polar-discriminator output carries a DC
    term from any residual tuning offset; left in, it biases the audio. State
    is carried across blocks so there's no click at block edges."""
    if x.size == 0:
        return x
    if state._dcblock_zi is None:
        state._dcblock_zi = (lfilter_zi([1.0, -1.0], [1.0, -r]) * float(x[0])).astype(np.float32)
    y, state._dcblock_zi = lfilter([1.0, -1.0], [1.0, -r], x, zi=state._dcblock_zi)
    return y.astype(np.float32)


@lru_cache(maxsize=8)
def _deemph_alpha(audio_rate: int, tau_us: float) -> float:
    """One-pole de-emphasis coefficient for time constant tau (microseconds).
    Region 1 (Europe) broadcast FM uses 50 us."""
    tau = tau_us * 1e-6
    dt = 1.0 / audio_rate
    return float(dt / (tau + dt))


def _deemphasis(audio: np.ndarray, audio_rate: int, tau_us: float = 50.0, state: DemodState = _default_state) -> np.ndarray:
    """Apply FM de-emphasis (matches the transmitter's pre-emphasis). Without
    this the high frequencies are boosted and FM sounds harsh/hissy. Stateful
    one-pole low-pass; scipy lfilter handles it efficiently and continuously."""
    if audio.size == 0:
        return audio
    a = _deemph_alpha(audio_rate, tau_us)
    # y[n] = a*x[n] + (1-a)*y[n-1]  ==  lfilter([a],[1,-(1-a)]) with state.
    zi = np.array([(1.0 - a) * state._deemph_prev], dtype=np.float32)
    y, zf = lfilter([a], [1.0, -(1.0 - a)], audio, zi=zi)
    state._deemph_prev = float(y[-1])
    return y.astype(np.float32)


# WBFM IF rate: 300 kHz → single decim=8 stage from 2.4 MHz capture.
# A SINGLE stage has passband = 0.85/8 × 1.2 MHz Nyquist = ±127.5 kHz,
# which is well above the ±90 kHz Carson bandwidth needed for broadcast FM
# (2 × (75 kHz dev + 15 kHz audio) = 180 kHz).
#
# The 200 kHz IF rate was WRONG: 2.4M / 200k = 12 = 4 × 3 → TWO stages,
# the second of which (decim=3) has passband = 0.85/3 × 300 kHz = ±85 kHz.
# That clips FM deviation → grainy/crushed audio when zoomed out.
#
# With 300 kHz the channelizer stage is clean, and a pre-demodulation complex
# lowpass (applied after channelization) rejects adjacent channels before the
# non-linear FM discriminator gets them.
WBFM_IF_RATE = 300_000


def demodulate_wbfm(
    iq: np.ndarray,
    *,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    audio_rate: int = DEFAULT_AUDIO_RATE,
    bandwidth_hz: int = 15_000,
    state: DemodState = _default_state,
) -> np.ndarray:
    iq_c = np.asarray(iq, dtype=np.complex64)
    # Channelize the IQ down to a narrow IF *before* the (non-linear)
    # discriminator. Without this the discriminator sees the whole captured band
    # -> adjacent stations + wideband noise fold into the audio, and the FM
    # capture effect locks onto whatever carrier is strongest (so clicking a few
    # kHz over still demods the old station). Decimating to a ~200 kHz IF first
    # isolates the wanted channel.
    iq_if, if_rate = _channelize(iq_c, sample_rate, WBFM_IF_RATE)

    # --- Pre-demodulation complex lowpass filter (adjacent-channel rejection) --
    # The channelizer output covers ±if_rate/2 (~±150 kHz at 300k IF). Adjacent
    # FM broadcast channels sit at ±100 kHz from the tuned carrier (100 kHz
    # raster in Europe). The FM discriminator is non-linear — if adjacent
    # channels reach it they create intermodulation distortion.
    #
    # Carson's rule gives the minimum IF bandwidth needed:
    #   BW = 2 × (Δf + f_max) = 2 × (75 000 + audio_bw) ≈ 180 kHz for 15 kHz audio
    # The wanted signal occupies ±90 kHz. We apply a steep complex lowpass with
    # cutoff at 95 kHz — this passes the full ±90 kHz Carson band cleanly while
    # putting the first adjacent channel at ±100 kHz well into the stopband
    # (a 255-tap Hamming FIR transitions in ~4 kHz at 300k sample rate).
    #
    # Note: 95 kHz may clip FM deviation >95 kHz which only matters for pirate
    # overdeviated signals; legal broadcast stays within 75 kHz nominal.
    carson_bw = int(2 * (75000 + bandwidth_hz))
    pre_bw = max(50000, min(if_rate // 2 - 10000, carson_bw))
    pre_bw = min(pre_bw, 95_000)
    if pre_bw < if_rate // 2 - 5000:
        # 511-tap Hamming FIR gives a very tight transition band (~2 kHz at
        # 300k IF rate) so the first adjacent FM channel at ±100 kHz is deep
        # in the stopband before the non-linear FM discriminator sees it.
        h_pre = _lowpass(if_rate, pre_bw, taps=511)
        iq_if = _filter_complex_stateful(h_pre, iq_if, state=state)

    # Measure signal power from the channelized NARROW IF, not the raw
    # full-band IQ. This gives a zoom-independent reading: the dBFS value
    # reflects power within the actual demodulated channel, so it doesn't
    # change when you zoom in/out.
    state._last_if_power_db = _signal_power_db(iq_if)

    # Polar-discriminator output is proportional to instantaneous frequency.
    audio_raw = _fm_demod(iq_if, state=state)
    # Normalize the discriminator output by the IF rate: the same FM deviation
    # produces a different angle at different IF rates (angle = 2π × Δf / fs).
    # At LOWER fs (deep zoom, 250k) the angle is LARGER, so we multiply by
    # if_rate / WBFM_IF_RATE (250/300 = 0.833× to quiet it, while at 300k it's
    # 300/300 = 1.0). This makes audio level the same at any zoom.
    audio_raw = audio_raw * (float(if_rate) / float(WBFM_IF_RATE))
    # Remove the DC term (residual tuning offset) at the IF rate so audio isn't
    # biased, then shape the FM audio channel at the IF rate.
    audio_dc = _dc_block(audio_raw, state=state)
    # Use configurable audio lowpass bandwidth (e.g., 15 kHz for full FM audio,
    # narrower for weak-signal NFM). Default 15 kHz drops 19 kHz pilot + stereo.
    bw = max(1_000, min(bandwidth_hz, if_rate // 2 - 1_000))
    audio_filt = _lfilter_stateful(_lowpass(if_rate, bw), audio_dc, state=state)
    # 50 us de-emphasis (Region 1), at the rate it actually runs at.
    audio_de = _deemphasis(audio_filt, if_rate, tau_us=50.0, state=state)
    # Now drop from the IF rate to the audio rate (resample_poly anti-aliases).
    audio_out = _decimate(audio_de, if_rate, audio_rate)
    # FM is constant-envelope, so loudness is set by deviation, not signal power.
    # A chasing AGC pumps the audio; use a fixed, soft-clipped gain instead.
    return _fm_gain(audio_out, state=state)


def _fm_gain(audio: np.ndarray, target: float = 0.25, state: DemodState = _default_state) -> np.ndarray:
    if audio.size == 0:
        return audio.astype(np.float32)
    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)))) + 1e-9
    # Fast enough to recover from a DSP-reset stale block (~1 s) but slow
    # enough not to pump on individual words (~50 ms syllable-rate floor).
    state._fm_level = 0.90 * state._fm_level + 0.10 * rms
    gain = min(20.0, target / max(state._fm_level, 1e-4))
    out = audio * gain
    # Gentle peak limiter instead of full-wave tanh. tanh(0.25) = 0.245 (nearly
    # linear), but tanh(1.0) = 0.76 — which is ~2.4 dB of compression that
    # sounds "crushed" on strong signals. We only limit samples that exceed a
    # moderate threshold (0.85), keeping clean signals untouched while catching
    # common transient peaks at ~3x the target RMS (0.25).
    threshold = 0.85
    above = np.abs(out)
    mask = above > threshold
    if np.any(mask):
        excess = above - threshold
        # Soft-clip the excess against the remaining headroom (1.0 - threshold)
        limited = threshold + (1.0 - threshold) * np.tanh(
            excess / max(1e-6, 1.0 - threshold)
        )
        out = np.where(mask, np.sign(out) * limited, out)
    return out.astype(np.float32)


# AM intermediate-frequency rate. An AM channel is ~10 kHz wide, so an IF at the
# audio rate (48 kHz) comfortably passes it while isolating the channel from the
# rest of the captured band before the envelope detector.
AM_IF_RATE = 48_000


def demodulate_am(
    iq: np.ndarray,
    *,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    audio_rate: int = DEFAULT_AUDIO_RATE,
    bandwidth_hz: int = 5_000,
    state: DemodState = _default_state,
) -> np.ndarray:
    # Channelize the IQ to a narrow IF first so the envelope detector only sees
    # the wanted AM channel, not the whole captured band (same root fix as WBFM).
    iq_if, if_rate = _channelize(np.asarray(iq, dtype=np.complex64), sample_rate, AM_IF_RATE, state=state)
    state._last_if_power_db = _signal_power_db(iq_if)
    envelope = np.abs(iq_if).astype(np.float32)
    envelope -= float(envelope.mean())
    audio_dec = _decimate(envelope, if_rate, audio_rate)
    bw = max(500, min(bandwidth_hz, audio_rate // 2 - 500))
    audio_filt = _lfilter_stateful(_lowpass(audio_rate, bw), audio_dec, state=state)
    return _agc(audio_filt, state=state)


# SSB/CW intermediate-frequency rate. The audio is at most ~3 kHz wide; an IF at
# the audio rate isolates it and lets the steep sideband filter be cheap+sharp
# (a 255-tap FIR against a 2.4 MHz Nyquist cannot cut at 3 kHz at all).
SSB_IF_RATE = 48_000


def demodulate_ssb(
    iq: np.ndarray,
    *,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    audio_rate: int = DEFAULT_AUDIO_RATE,
    sideband: str = "usb",
    bandwidth_hz: int = 3_000,
    state: DemodState = _default_state,
) -> np.ndarray:
    # Channelize to a low IF first so the sideband filter operates at a rate
    # where it can actually be steep.
    samples, if_rate = _channelize(np.asarray(iq, dtype=np.complex64), sample_rate, SSB_IF_RATE, state=state)
    state._last_if_power_db = _signal_power_db(samples)
    if sideband.lower() == "lsb":
        samples = np.conj(samples)
    taps = _lowpass(if_rate, bandwidth_hz, taps=255)
    key = id(taps)
    zi = state._filter_state.get(key)
    if zi is None or zi.shape != (len(taps) - 1,):
        zi = np.zeros(len(taps) - 1, dtype=np.complex64)
    filtered, zf = lfilter(taps, [1.0], samples, zi=zi)
    state._filter_state[key] = zf
    audio_dec = _decimate(filtered.real.astype(np.float32), if_rate, audio_rate)
    return _agc(audio_dec, state=state)




def _frequency_shift(iq: np.ndarray, freq_offset_hz: float, sample_rate: int, state: DemodState = _default_state) -> np.ndarray:
    """Digitally down-convert the tuned signal to baseband.

    The hardware captures center_freq +/- sample_rate/2. The signal the user
    actually wants sits at ``freq_offset_hz`` from that centre. Multiplying by
    exp(-j*2*pi*offset/fs*n) slides that signal down to DC so the existing
    demod chain (which assumes the signal is at baseband) works unchanged.

    Phase is accumulated across blocks so there is no discontinuity at block
    boundaries; we also reset the running phase whenever the offset changes so
    a retune doesn't carry a stale phase.
    """
    if freq_offset_hz == 0.0 or iq.size == 0:
        return iq
    if freq_offset_hz != state._tune_offset_active:
        state._tune_phase = 0.0
        state._tune_offset_active = freq_offset_hz
    n = np.arange(iq.size, dtype=np.float64)
    w = 2.0 * np.pi * (freq_offset_hz / float(sample_rate))
    phase = state._tune_phase + w * n
    mixer = np.exp(-1j * phase).astype(np.complex64)
    # Advance the running phase, wrapped to keep it bounded.
    state._tune_phase = float((state._tune_phase + w * iq.size) % (2.0 * np.pi))
    return (np.asarray(iq, dtype=np.complex64) * mixer).astype(np.complex64)


# ---------------------------------------------------------------------------
# Signal power measurement
# ---------------------------------------------------------------------------

def _signal_power_db(iq: np.ndarray) -> float:
    """RMS signal power in dBFS of complex IQ samples.

    Returns a float in dBFS (0 dBFS = full scale of the 8-bit ADC).
    This is the raw power of the channelized IF signal *before* demodulation,
    used by the squelch system and signal strength meter.
    """
    if iq.size == 0:
        return -160.0
    power = float(np.mean(np.real(iq) ** 2 + np.imag(iq) ** 2))
    eps = max(power, 1e-18)
    return float(10.0 * np.log10(eps))


# ---------------------------------------------------------------------------
# Squelch with hysteresis
# ---------------------------------------------------------------------------


def squelch(
    audio: np.ndarray,
    signal_power_db: float,
    threshold_db: float,
    hysteresis_db: float = 3.0,
    state: DemodState = _default_state,
) -> np.ndarray:
    """Apply squelch with hysteresis.

    When the signal power is above ``threshold_db`` the squelch OPENS
    (passes audio). Once open, it stays open until the signal drops more
    than ``hysteresis_db`` below the threshold (prevents rapid
    open/close chattering around the threshold).

    Args:
        audio: PCM audio samples to gate.
        signal_power_db: Current signal power in dBFS (from _signal_power_db).
        threshold_db: Squelch threshold in dBFS. Set to -200 to disable.
        hysteresis_db: Hysteresis window in dB. Default 3 dB.

    Returns:
        audio (gated to zero when squelch is closed) if threshold is valid,
        or the original audio unchanged if threshold is effectively disabled.
    """

    # Squelch disabled: always pass audio, keep state open.
    if threshold_db <= -150.0:
        state._squelch_open = True
        return audio

    if audio.size == 0:
        return audio

    # Hysteresis decision:
    #   Opening: signal must rise ABOVE threshold_db.
    #   Closing: signal must fall BELOW (threshold_db - hysteresis_db).
    if state._squelch_open:
        # Currently open: close only if signal drops below the hysteresis floor.
        if signal_power_db < (threshold_db - hysteresis_db):
            state._squelch_open = False
    else:
        # Currently closed: open only if signal rises above the threshold.
        if signal_power_db > threshold_db:
            state._squelch_open = True

    if not state._squelch_open:
        # Mute audio with a brief linear fade to avoid a hard click when the
        # squelch closes mid-block.
        if audio.size > 0:
            fade_len = min(48, audio.size)  # 1 ms at 48 kHz
            for i in range(fade_len):
                audio[i] = audio[i] * (1.0 - float(i) / fade_len)
        return audio * 0.0

    return audio



def squelch_active(state: DemodState = _default_state) -> bool:
    """Return the current squelch open/closed state."""
    return state._squelch_open


def last_signal_power_db(state: DemodState = _default_state) -> float:
    """Return the most recently measured signal power in dBFS."""
    return state._last_signal_power_db


# ---------------------------------------------------------------------------
# Mute transient after DSP state reset — prevents clicks on zoom/rate change
# ---------------------------------------------------------------------------


def _apply_mute_ramp(audio: np.ndarray, state: DemodState = _default_state) -> np.ndarray:
    """Apply a fade-in gain ramp to suppress clicks after DSP state reset.

    The first ``ramp_len`` samples of the output are linearly faded in from
    silence to full gain. This masks the filter transient and phase
    discontinuity that happens when _reset_audio_dsp_state() clears all
    filter histories.

    Args:
        audio: PCM audio block (float32).
        ramp_len: Number of samples to fade in (default 240 = 5 ms at 48 kHz).

    Returns:
        Audio with the leading edge faded in.
    """
    if state._mute_ramp <= 0 or audio.size == 0:
        return audio

    fade = min(state._mute_ramp, audio.size)
    for i in range(fade):
        audio[i] = audio[i] * float(i) / fade
    state._mute_ramp -= fade
    return audio


# ---------------------------------------------------------------------------
# Demodulation dispatcher
# ---------------------------------------------------------------------------


def demodulate(
    iq: np.ndarray,
    mode: str,
    *,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    audio_rate: int = DEFAULT_AUDIO_RATE,
    freq_offset_hz: float = 0.0,
    bandwidth_hz: int | None = None,
    squelch_threshold: float = -200.0,
    state: DemodState = _default_state,
) -> np.ndarray:
    mode = mode.lower()

    if sample_rate != state._last_demod_rate:
        _reset_audio_dsp_state(state)
        state._last_demod_rate = sample_rate
        # Start mute ramp to suppress filter startup transient AND any stale IQ
        # block that was captured at the old rate (ring.clear() catches most, but
        # a race can leave one block). Scaled to ~60 ms at the current audio rate,
        # which covers one full block at the worst case (0.25 MHz → ~262 ms block,
        # but the per-block delivery is pipelined so ~60 ms is safe without audible
        # silence gaps).
        state._mute_ramp = int(0.060 * max(audio_rate, 1))
    elif freq_offset_hz != state._last_tune_offset:
        # Every click-to-tune changes the offset. Even a small nearby click
        # means the current IQ block was captured at the OLD offset -- stale
        # data that would produce a brief burst of wrong-frequency audio.
        # Always apply a mute ramp; scale by how far we moved.
        if abs(freq_offset_hz - state._last_tune_offset) > 50_000:
            # Big retune/zoom-induced jump while rate stayed the same:
            # clear AGC / FM leveller so the new signal doesn't inherit
            # pumping or level from the previous one.
            state._agc_level = 1e-3
            state._agc_gain = 1.0
            state._fm_level = 0.25
            state._mute_ramp = int(0.010 * max(audio_rate, 1))  # ~10 ms
        else:
            # Small nearby click: short 5 ms ramp masks the stale-block
            # blurp without audible silence.
            state._mute_ramp = int(0.005 * max(audio_rate, 1))  # ~5 ms

    state._last_tune_offset = freq_offset_hz

    # Slide the tuned signal down to baseband before demodulating. When the
    # user clicks off-centre on the waterfall the band stays put, so the signal
    # is no longer at DC -- this shift brings it there.
    iq_shifted = _frequency_shift(
        np.asarray(iq, dtype=np.complex64), freq_offset_hz, sample_rate, state=state
    )

    # Each demod function sets _last_if_power_db from its channelized IF
    # (zoom-independent reading). The line below was a dead store — always
    # immediately overwritten by the demod's own measurement. Removed.

    # Resolve bandwidth: use mode defaults if not specified.
    if bandwidth_hz is None:
        bw_map = {"wbfm": 15_000, "am": 5_000, "usb": 3_000, "lsb": 3_000, "cw": 900}
        bw = bw_map.get(mode, 5_000)
    else:
        bw = int(bandwidth_hz)

    if mode == "wbfm":
        pcm = demodulate_wbfm(
            iq_shifted, sample_rate=sample_rate, audio_rate=audio_rate, bandwidth_hz=bw, state=state
        )
    elif mode == "am":
        pcm = demodulate_am(
            iq_shifted, sample_rate=sample_rate, audio_rate=audio_rate, bandwidth_hz=bw, state=state
        )
    elif mode in {"usb", "lsb", "cw"}:
        sideband = "usb" if mode == "cw" else mode
        pcm = demodulate_ssb(
            iq_shifted,
            sample_rate=sample_rate,
            audio_rate=audio_rate,
            sideband=sideband,
            bandwidth_hz=bw,
            state=state,
        )
    else:
        raise ValueError(f"unsupported demodulation mode: {mode}")

    # Use the channelized IF power (set inside each demod function). This is
    # zoom-independent: it reflects power within the actual demod bandwidth,
    # not the full captured band, so the signal meter isn't misleading.
    state._last_signal_power_db = state._last_if_power_db

    # Apply mute ramp to suppress DSP reset clicks
    pcm = _apply_mute_ramp(pcm, state=state)

    # Apply squelch using the zoom-independent IF signal power
    pcm = squelch(pcm, state._last_signal_power_db, float(squelch_threshold), state=state)

    return pcm
