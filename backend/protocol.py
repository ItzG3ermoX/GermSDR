from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

from .state import RadioConfig


# [uint32 seq][float64 center_freq][float32 sample_rate][uint16 fft_size][uint16 flags]
# The 20-byte header keeps the Float32 bin payload 4-byte aligned in browsers.
HEADER_FMT = "!IdfHH"
HEADER_SIZE = struct.calcsize(HEADER_FMT)
FLAG_PEAK_HOLD = 1 << 0


@dataclass(frozen=True)
class WaterfallHeader:
    seq: int
    center_freq: float
    sample_rate: float
    fft_size: int
    flags: int = 0


def make_waterfall_frame(seq: int, config: RadioConfig, bins: np.ndarray, flags: int = 0) -> bytes:
    payload = np.asarray(bins, dtype="<f4")
    header = struct.pack(
        HEADER_FMT,
        seq & 0xFFFFFFFF,
        float(config.center_freq),
        float(config.sample_rate),
        int(payload.size),
        flags,
    )
    return header + payload.tobytes(order="C")


def parse_waterfall_header(frame: bytes) -> WaterfallHeader:
    seq, center_freq, sample_rate, fft_size, flags = struct.unpack(
        HEADER_FMT, frame[:HEADER_SIZE]
    )
    return WaterfallHeader(seq, center_freq, sample_rate, fft_size, flags)

