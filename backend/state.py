from __future__ import annotations

from dataclasses import asdict, dataclass
from threading import RLock


VALID_MODES = {"wbfm", "am", "usb", "lsb", "cw"}
VALID_FFT_SIZES = {2_048, 4_096, 8_192, 16_384, 32_768}

# RTL-SDR (R820T/R828D) clean native sample rates, descending. The chip can only
# sample reliably in 225001-300000 Hz and 900001-3200000 Hz; the 300k-900k gap
# drops samples, so we step through these proven-stable rates only. The first
# entry is the reference ("base") rate that perceived zoom is measured against.
HW_RATE_LADDER = (2_400_000, 1_200_000, 960_000, 250_000)
BASE_RATE = HW_RATE_LADDER[0]

# Keep the residual DIGITAL zoom in this range. When perceived zoom would push
# the digital factor past the max, we drop to the next narrower hardware rate so
# the tuner delivers the resolution natively (sharp, cheap) instead of the DSP
# decimating a huge band. Min 1.0 because we cannot digitally zoom OUT past the
# captured band.
#
# Set at 4.0 so small waterfall zooms (up to 4x, showing 600 kHz) stay purely
# digital on the 2.4 MHz rate — a single wheel tick shouldn't retune the
# hardware and collapse the entire band. Only when zooming deeper (5x+) do we
# step down the hardware rate ladder for sharper resolution and cleaner audio.
DIGITAL_ZOOM_MIN = 1.0
DIGITAL_ZOOM_MAX = 4.0

# Demodulation bandwidth presets (Hz) for each mode.
# Format: {"min": min_hz, "max": max_hz, "default": default_hz, "step": step_hz}
BANDWIDTH_PRESETS: dict[str, dict[str, int]] = {
    "wbfm": {"min": 5_000, "max": 250_000, "default": 15_000, "step": 1_000},
    "am":   {"min": 1_000, "max": 20_000,  "default": 5_000,  "step": 500},
    "usb":  {"min": 200,   "max": 8_000,   "default": 3_000,  "step": 100},
    "lsb":  {"min": 200,   "max": 8_000,   "default": 3_000,  "step": 100},
    "cw":   {"min": 50,    "max": 2_000,   "default": 500,    "step": 50},
}

# Squelch range: -100 dBFS (strong signal) to -20 dBFS (very strong),
# or -160 to disable.
SQUELCH_MIN_DB = -160.0
SQUELCH_MAX_DB = -20.0


def default_bandwidth(mode: str) -> int:
    """Return the default bandwidth in Hz for a given mode."""
    p = BANDWIDTH_PRESETS.get(mode.lower(), BANDWIDTH_PRESETS["am"])
    return p["default"]


def hw_rate_for_zoom(perceived_zoom: float) -> int:
    """Pick the widest ladder rate whose required digital zoom stays <= MAX.

    Perceived zoom is relative to BASE_RATE: visible span = BASE_RATE/zoom. At a
    hardware rate R the captured band is R wide, so the residual digital zoom
    needed is zoom * R / BASE_RATE. We want the WIDEST R (least retuning, most
    band visible for audio) that keeps that residual <= DIGITAL_ZOOM_MAX."""
    z = max(1.0, float(perceived_zoom))
    chosen = HW_RATE_LADDER[-1]
    for rate in HW_RATE_LADDER:  # widest first
        if z * rate / BASE_RATE <= DIGITAL_ZOOM_MAX:
            chosen = rate
            break
    else:
        chosen = HW_RATE_LADDER[-1]
    return chosen


def digital_zoom_for(perceived_zoom: float, hw_rate: int) -> float:
    """Residual digital zoom to apply on top of a hardware rate so the visible
    span equals BASE_RATE/perceived_zoom."""
    z = max(1.0, float(perceived_zoom)) * float(hw_rate) / float(BASE_RATE)
    return max(DIGITAL_ZOOM_MIN, z)


@dataclass(frozen=True)
class RadioConfig:
    center_freq: int
    sample_rate: int
    gain: str
    mode: str = "wbfm"
    ppm: float = 0.0
    fft_size: int = 32_768
    demod_bw: int = 15_000  # demodulation bandwidth in Hz
    squelch: float = -160.0  # squelch threshold in dBFS (-160 = off)
    rtl_agc: bool = True  # RTL-SDR digital AGC: protects the 8-bit ADC from
                            # overload at wide RF bandwidths by dynamically
                            # adjusting the ADC reference level. Keeps the ADC
                            # out of clipping when the R828D tuner is set to
                            # 2.4 MHz (wider IF filter → more noise → higher
                            # peak-to-average power at the ADC input).
    bias_tee: bool = False  # Bias-T enable
    direct_sampling: int = 0  # 0=off, 1=I, 2=Q

    def as_dict(self) -> dict[str, int | float | str | bool]:
        return asdict(self)


class RadioState:
    """Thread-safe tuning state shared by capture and websocket tasks."""

    def __init__(self, config: RadioConfig):
        self._validate_center_freq(config.center_freq)
        self._config = config
        self._lock = RLock()
        # Display-only view state. Kept OUT of RadioConfig so that scrolling the
        # waterfall never triggers a hardware reconfigure in the capture thread.
        # Perceived zoom is what the user dialed in (relative to BASE_RATE);
        # _zoom is the RESIDUAL digital zoom the DSP applies on top of whatever
        # hardware rate we auto-selected for that perceived zoom.
        self._perceived_zoom: float = 1.0
        self._zoom: float = 1.0
        # Where, inside the captured band, the user is actually listening. This
        # is the Hz offset of the tuned signal from the hardware centre_freq.
        # Click-to-tune sets this WITHOUT retuning hardware, so the waterfall
        # band stays put while the demodulator + zoom follow the clicked signal.
        # 0 = listening at band centre (the old always-centred behaviour).
        self._tune_offset_hz: float = 0.0
        # Pan is the centre of the visible (zoomed) window across the captured
        # band, 0..1 (0.5 = band centre). It is INDEPENDENT of the tune offset:
        # the zoom window stays where the user put it, and clicking inside it
        # only moves the listening point (marker), not the view. The window only
        # follows the tuned signal when that signal reaches the window edge.
        self._pan: float = 0.5

    def snapshot(self) -> RadioConfig:
        with self._lock:
            return self._config

    @property
    def tune_offset(self) -> float:
        """Hz offset of the tuned signal from the hardware centre frequency."""
        with self._lock:
            return self._tune_offset_hz

    @property
    def tuned_freq(self) -> float:
        """Absolute frequency (Hz) the user is currently listening to."""
        with self._lock:
            return float(self._config.center_freq) + self._tune_offset_hz

    @property
    def view(self) -> tuple[float, float]:
        """Current (zoom, pan) display window. Pan is an independent view
        position (NOT derived from the tune offset)."""
        with self._lock:
            return self._zoom, self._pan

    def set_view(
        self,
        zoom: float | None = None,
        pan: float | None = None,
        pan_hz: float | None = None,
    ) -> None:
        """Update the view. Zoom is PERCEIVED zoom (relative to BASE_RATE). When
        it crosses a hardware-rate threshold we retune the tuner to a narrower
        native rate centred on the tuned signal -- giving real ADC resolution on
        that band (sharp + no DSP lag) -- and apply only the leftover digital
        zoom. The visible span (BASE_RATE/zoom) stays continuous across switches.

        ``pan`` is the view centre as a 0..1 fraction of the captured band.
        ``pan_hz`` is the same thing expressed as an ABSOLUTE frequency (Hz) --
        used by drag-to-pan, which works in real frequency (it knows the on-screen
        slice's centre/rate, not the captured-band fraction). We convert pan_hz to
        the band fraction here, where the authoritative band centre/rate live.
        """
        with self._lock:
            if zoom is not None:
                self._perceived_zoom = max(1.0, min(64.0, float(zoom)))
                target_rate = hw_rate_for_zoom(self._perceived_zoom)
                cur_rate = int(self._config.sample_rate)

                if target_rate != cur_rate:
                    # Retune the hardware: centre the narrower band on the signal
                    # the user is listening to so audio keeps working, then reset
                    # the listening point (now at band centre) and the pan.
                    new_center = int(round(self.tuned_freq))
                    self._config = RadioConfig(
                        center_freq=new_center,
                        sample_rate=target_rate,
                        gain=self._config.gain,
                        mode=self._config.mode,
                        ppm=self._config.ppm,
                        fft_size=self._config.fft_size,
                        demod_bw=self._config.demod_bw,
                        squelch=self._config.squelch,
                        rtl_agc=self._config.rtl_agc,
                        bias_tee=self._config.bias_tee,
                        direct_sampling=self._config.direct_sampling,
                    )
                    self._tune_offset_hz = 0.0
                    self._pan = 0.5

                # Residual digital zoom on top of whatever hardware rate is now set.
                self._zoom = digital_zoom_for(self._perceived_zoom, int(self._config.sample_rate))

            # Apply pan AFTER any zoom/rate switch so a drag-release pan is never
            # clobbered by the switch's pan reset. pan_hz (absolute) takes
            # precedence over the 0..1 form when both are sent.
            if pan_hz is not None:
                self._apply_pan_hz(float(pan_hz))
            elif pan is not None:
                self._pan = min(1.0, max(0.0, float(pan)))

    # Allowed hardware tuning range (Hz). The RTL-SDR/R820T tunes roughly
    # 24 MHz - 1.766 GHz; we keep a generous clamp so a drag can scroll the band
    # without running off into an invalid tune.
    TUNE_MIN_HZ = 24_000_000
    TUNE_MAX_HZ = 1_766_000_000

    @classmethod
    def _validate_center_freq(cls, center_freq: int) -> None:
        if not cls.TUNE_MIN_HZ <= center_freq <= cls.TUNE_MAX_HZ:
            raise ValueError(
                f"center_freq must be between {cls.TUNE_MIN_HZ} and {cls.TUNE_MAX_HZ} Hz"
            )

    def _apply_pan_hz(self, pan_hz: float) -> None:
        """Set the view centre to an absolute frequency.

        While there is room to pan *within* the captured band (zoomed in, so the
        visible window is narrower than the band) we just move the view fraction.
        Once the requested centre would push the visible window past a band edge
        -- which is always the case at 1x, where the whole band is already on
        screen -- there is nothing left to pan to, so we RETUNE the hardware
        centre to follow the drag and scroll the whole band (KiwiSDR-style).
        """
        rate = float(self._config.sample_rate) or 1.0
        center = float(self._config.center_freq)
        # Half-width of the visible window as a fraction of the captured band.
        half_span = 0.5 / max(1.0, self._zoom)
        # The view centre can range over [half_span, 1 - half_span] before the
        # window hits a band edge. Convert the requested centre to that fraction.
        frac = 0.5 + (pan_hz - center) / rate

        # There is room to pan within the captured band only when the visible
        # window is narrower than the band itself (i.e. we are zoomed in). When
        # zoomed in, ALWAYS keep the pan inside the band by clamping to the edge
        # -- never retune the hardware -- so a drag that runs past the edge just
        # parks at the edge instead of teleporting the centre. Retuning is for
        # 1x only, where the whole band is already on screen and there is nothing
        # left to scroll without moving the hardware.
        pannable = half_span < 0.5  # strictly zoomed in
        if pannable:
            self._pan = min(1.0 - half_span, max(half_span, frac))
            return

        # 1x (no in-band headroom): scroll the band by retuning the hardware
        # centre to follow the drag (KiwiSDR-style). Cap the step to one captured
        # band width per pan so a stray/garbage pan_hz can never teleport the
        # tuner across the spectrum (this was the "+/-25 MHz jump").
        max_step = rate
        target = min(center + max_step, max(center - max_step, pan_hz))
        new_center = int(round(min(self.TUNE_MAX_HZ, max(self.TUNE_MIN_HZ, target))))
        if new_center != int(self._config.center_freq):
            self._config = RadioConfig(
                center_freq=new_center,
                sample_rate=self._config.sample_rate,
                gain=self._config.gain,
                mode=self._config.mode,
                ppm=self._config.ppm,
                fft_size=self._config.fft_size,
                demod_bw=self._config.demod_bw,
                squelch=self._config.squelch,
                rtl_agc=self._config.rtl_agc,
                bias_tee=self._config.bias_tee,
                direct_sampling=self._config.direct_sampling,
            )
            self._tune_offset_hz = 0.0
        self._pan = 0.5

    def _offset_to_pan(self, offset_hz: float) -> float:
        rate = float(self._config.sample_rate) or 1.0
        return 0.5 + offset_hz / rate

    def set_tune_offset(self, offset_hz: float) -> None:
        """Set where in the captured band we listen, clamped to the band edges.

        The zoom window (pan) does NOT recentre on the new point unless that
        point would fall outside the currently visible window -- then the window
        follows so the tuned signal stays reachable (KiwiSDR-style edge-follow).
        """
        with self._lock:
            half = float(self._config.sample_rate) / 2.0
            self._tune_offset_hz = max(-half, min(half, float(offset_hz)))
            # Where does the tuned point now sit within the visible window?
            tuned_pan = self._offset_to_pan(self._tune_offset_hz)
            half_span = 0.5 / max(1.0, self._zoom)
            lo = self._pan - half_span
            hi = self._pan + half_span
            # Only move the view if the tuned point left the visible window.
            if tuned_pan < lo:
                self._pan = min(1.0, max(0.0, tuned_pan + half_span * 0.9))
            elif tuned_pan > hi:
                self._pan = min(1.0, max(0.0, tuned_pan - half_span * 0.9))

    def update(
        self,
        *,
        center_freq: int | None = None,
        mode: str | None = None,
        gain: str | float | None = None,
        ppm: float | None = None,
        fft_size: int | None = None,
        demod_bw: int | None = None,
        squelch: float | None = None,
        rtl_agc: bool | None = None,
        bias_tee: bool | None = None,
        direct_sampling: int | None = None,
    ) -> RadioConfig:
        with self._lock:
            if center_freq is not None:
                self._validate_center_freq(center_freq)
            next_mode = (mode or self._config.mode).lower()
            if next_mode not in VALID_MODES:
                allowed = ", ".join(sorted(VALID_MODES))
                raise ValueError(f"mode must be one of: {allowed}")

            next_fft = fft_size or self._config.fft_size
            if next_fft not in VALID_FFT_SIZES:
                allowed = ", ".join(str(v) for v in sorted(VALID_FFT_SIZES))
                raise ValueError(f"fft_size must be one of: {allowed}")

            next_gain = self._config.gain if gain is None else str(gain)

            # Demod bandwidth: if mode changed, auto-select default for new mode.
            # If explicitly provided, use it (clamped to mode presets).
            if demod_bw is not None:
                next_bw = int(demod_bw)
            elif mode is not None and mode.lower() != self._config.mode:
                next_bw = default_bandwidth(next_mode)
            else:
                next_bw = self._config.demod_bw

            # Squelch threshold: -160 = off, -100...-20 = active range
            next_squelch = self._config.squelch if squelch is None else float(squelch)

            # RTL-SDR advanced controls
            next_rtl_agc = self._config.rtl_agc if rtl_agc is None else bool(rtl_agc)
            next_bias_tee = self._config.bias_tee if bias_tee is None else bool(bias_tee)
            next_direct_sampling = (
                self._config.direct_sampling
                if direct_sampling is None
                else int(direct_sampling)
            )

            # A hardware retune (centre_freq change) makes the new frequency the
            # band centre, so the listening point is back at offset 0.
            if center_freq is not None and center_freq != self._config.center_freq:
                self._tune_offset_hz = 0.0
                self._pan = 0.5
            self._config = RadioConfig(
                center_freq=self._config.center_freq if center_freq is None else center_freq,
                sample_rate=self._config.sample_rate,
                gain=next_gain,
                mode=next_mode,
                ppm=self._config.ppm if ppm is None else ppm,
                fft_size=next_fft,
                demod_bw=next_bw,
                squelch=next_squelch,
                rtl_agc=next_rtl_agc,
                bias_tee=next_bias_tee,
                direct_sampling=next_direct_sampling,
            )
            return self._config
