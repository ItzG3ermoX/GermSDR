from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return int(value.replace("_", ""))


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return float(value)


@dataclass(frozen=True)
class Settings:
    sample_rate: int = 2_400_000
    center_freq: int = 100_800_000
    gain: str = "auto"
    ppm: float = 0.0
    fft_size: int = 32_768
    block_size: int = 65_536
    ring_slots: int = 16
    waterfall_fps: float = 25.0
    audio_rate: int = 48_000
    source: str = "auto"
    host: str = "0.0.0.0"
    port: int = 8080
    frontend_dist: Path = Path(__file__).resolve().parents[1] / "frontend" / "dist"

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            sample_rate=_env_int("SDR_RATE", cls.sample_rate),
            center_freq=_env_int("SDR_FREQ", cls.center_freq),
            gain=os.getenv("SDR_GAIN", cls.gain),
            ppm=_env_float("SDR_PPM", cls.ppm),
            fft_size=_env_int("SDR_FFT", cls.fft_size),
            block_size=_env_int("SDR_BLOCK", cls.block_size),
            ring_slots=_env_int("SDR_RING_SLOTS", cls.ring_slots),
            waterfall_fps=_env_float("SDR_WATERFALL_FPS", cls.waterfall_fps),
            audio_rate=_env_int("SDR_AUDIO_RATE", cls.audio_rate),
            source=os.getenv("SDR_SOURCE", cls.source).lower(),
            host=os.getenv("SDR_HOST", cls.host),
            port=_env_int("SDR_PORT", cls.port),
            frontend_dist=Path(os.getenv("SDR_FRONTEND_DIST", str(cls.frontend_dist))),
        )

