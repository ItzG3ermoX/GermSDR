from __future__ import annotations

from dataclasses import asdict, dataclass
from threading import RLock


VALID_MODES = {"wbfm", "am", "usb", "lsb", "cw"}
VALID_FFT_SIZES = {2_048, 4_096, 8_192, 16_384, 32_768}


@dataclass(frozen=True)
class RadioConfig:
    center_freq: int
    sample_rate: int
    gain: str
    mode: str = "wbfm"
    ppm: float = 0.0
    fft_size: int = 32_768

    def as_dict(self) -> dict[str, int | float | str]:
        return asdict(self)


class RadioState:
    """Thread-safe tuning state shared by capture and websocket tasks."""

    def __init__(self, config: RadioConfig):
        self._config = config
        self._lock = RLock()

    def snapshot(self) -> RadioConfig:
        with self._lock:
            return self._config

    def update(
        self,
        *,
        center_freq: int | None = None,
        mode: str | None = None,
        gain: str | float | None = None,
        ppm: float | None = None,
        fft_size: int | None = None,
    ) -> RadioConfig:
        with self._lock:
            next_mode = (mode or self._config.mode).lower()
            if next_mode not in VALID_MODES:
                allowed = ", ".join(sorted(VALID_MODES))
                raise ValueError(f"mode must be one of: {allowed}")

            next_fft = fft_size or self._config.fft_size
            if next_fft not in VALID_FFT_SIZES:
                allowed = ", ".join(str(v) for v in sorted(VALID_FFT_SIZES))
                raise ValueError(f"fft_size must be one of: {allowed}")

            next_gain = self._config.gain if gain is None else str(gain)
            self._config = RadioConfig(
                center_freq=self._config.center_freq if center_freq is None else center_freq,
                sample_rate=self._config.sample_rate,
                gain=next_gain,
                mode=next_mode,
                ppm=self._config.ppm if ppm is None else ppm,
                fft_size=next_fft,
            )
            return self._config

