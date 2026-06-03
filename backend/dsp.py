from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.signal import firwin, lfilter

try:
    from scipy.signal.windows import blackmanharris
except ImportError:  # pragma: no cover - for older scipy builds
    blackmanharris = None

try:
    from numba import njit
except ImportError:  # Python 3.14 often arrives before numba wheels do.
    njit = None


DEFAULT_SAMPLE_RATE = 2_400_000
DEFAULT_AUDIO_RATE = 48_000


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
    """Return fft_size float32 bins in dBFS, shifted so DC is centered."""
    samples = np.asarray(iq, dtype=np.complex64)
    if samples.size < fft_size:
        segment = np.zeros(fft_size, dtype=np.complex64)
        segment[: samples.size] = samples
    else:
        segment = samples[:fft_size]

    spectrum = np.fft.fftshift(np.fft.fft(segment * _window(fft_size, window), fft_size))
    power_db = 20.0 * np.log10(np.abs(spectrum) / fft_size + 1e-12)
    return power_db.astype(np.float32)


if njit is not None:

    @njit(cache=True)
    def _fm_demod(iq: np.ndarray) -> np.ndarray:
        n = len(iq)
        out = np.empty(n - 1, dtype=np.float32)
        for i in range(n - 1):
            z = iq[i + 1] * np.conj(iq[i])
            out[i] = np.arctan2(z.imag, z.real)
        return out

else:

    def _fm_demod(iq: np.ndarray) -> np.ndarray:
        return np.angle(iq[1:] * np.conj(iq[:-1])).astype(np.float32)


@lru_cache(maxsize=32)
def _lowpass(sample_rate: int, cutoff_hz: int, taps: int = 127) -> np.ndarray:
    nyquist = sample_rate / 2.0
    normalized = min(0.99, cutoff_hz / nyquist)
    return firwin(taps, normalized, window="hamming").astype(np.float32)


def _normalize(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak < 1e-9:
        return audio
    return (audio / peak).astype(np.float32)


def _decimation(sample_rate: int, audio_rate: int) -> int:
    return max(1, round(sample_rate / audio_rate))


def demodulate_wbfm(
    iq: np.ndarray,
    *,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    audio_rate: int = DEFAULT_AUDIO_RATE,
) -> np.ndarray:
    audio_raw = _fm_demod(np.asarray(iq, dtype=np.complex64))
    audio_filt = lfilter(_lowpass(sample_rate, 15_000), [1.0], audio_raw)
    audio_dec = audio_filt[::_decimation(sample_rate, audio_rate)]
    return _normalize(audio_dec)


def demodulate_am(
    iq: np.ndarray,
    *,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    audio_rate: int = DEFAULT_AUDIO_RATE,
) -> np.ndarray:
    envelope = np.abs(np.asarray(iq, dtype=np.complex64)).astype(np.float32)
    envelope -= float(envelope.mean())
    audio_filt = lfilter(_lowpass(sample_rate, 5_000), [1.0], envelope)
    return _normalize(audio_filt[::_decimation(sample_rate, audio_rate)])


def demodulate_ssb(
    iq: np.ndarray,
    *,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    audio_rate: int = DEFAULT_AUDIO_RATE,
    sideband: str = "usb",
    bandwidth_hz: int = 3_000,
) -> np.ndarray:
    samples = np.asarray(iq, dtype=np.complex64)
    if sideband.lower() == "lsb":
        samples = np.conj(samples)
    filtered = lfilter(_lowpass(sample_rate, bandwidth_hz, taps=255), [1.0], samples)
    return _normalize(filtered.real[::_decimation(sample_rate, audio_rate)])


def demodulate(
    iq: np.ndarray,
    mode: str,
    *,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    audio_rate: int = DEFAULT_AUDIO_RATE,
) -> np.ndarray:
    mode = mode.lower()
    if mode == "wbfm":
        return demodulate_wbfm(iq, sample_rate=sample_rate, audio_rate=audio_rate)
    if mode == "am":
        return demodulate_am(iq, sample_rate=sample_rate, audio_rate=audio_rate)
    if mode in {"usb", "lsb", "cw"}:
        sideband = "usb" if mode == "cw" else mode
        return demodulate_ssb(
            iq,
            sample_rate=sample_rate,
            audio_rate=audio_rate,
            sideband=sideband,
            bandwidth_hz=900 if mode == "cw" else 3_000,
        )
    raise ValueError(f"unsupported demodulation mode: {mode}")

