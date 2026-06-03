from __future__ import annotations

import logging
import math
import threading
import time
from typing import Protocol

import numpy as np

from .config import Settings
from .state import RadioState


LOGGER = logging.getLogger(__name__)


class IQRingBuffer:
    """Bounded single-producer/single-consumer IQ ring.

    The producer drops stale unread blocks when the ring laps the consumer. A
    short lock protects sequence counters and slot copies across Python threads.
    """

    def __init__(self, slots: int, block_size: int):
        self.slots = slots
        self.block_size = block_size
        self._buffer = np.zeros((slots, block_size), dtype=np.complex64)
        self._write_seq = 0
        self._read_seq = 0
        self._lock = threading.Lock()
        self.dropped_blocks = 0

    @property
    def depth(self) -> int:
        with self._lock:
            return self._write_seq - self._read_seq

    def push(self, samples: np.ndarray) -> None:
        block = np.asarray(samples, dtype=np.complex64)
        if block.size != self.block_size:
            fixed = np.zeros(self.block_size, dtype=np.complex64)
            fixed[: min(block.size, self.block_size)] = block[: self.block_size]
            block = fixed

        with self._lock:
            if self._write_seq - self._read_seq >= self.slots:
                self._read_seq = self._write_seq - self.slots + 1
                self.dropped_blocks += 1

            self._buffer[self._write_seq % self.slots] = block
            self._write_seq += 1

    def pop(self) -> np.ndarray | None:
        with self._lock:
            if self._read_seq == self._write_seq:
                return None

            out = self._buffer[self._read_seq % self.slots].copy()
            self._read_seq += 1
            return out


class CaptureSource(Protocol):
    name: str

    def start(self) -> None: ...

    def stop(self) -> None: ...


class SimulatedCaptureSource:
    name = "simulated"

    def __init__(self, ring: IQRingBuffer, state: RadioState, settings: Settings):
        self._ring = ring
        self._state = state
        self._settings = settings
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._rng = np.random.default_rng(7)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="sdr-sim-capture", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.5)

    def _run(self) -> None:
        n0 = 0
        phase = 0.0
        block = self._settings.block_size

        while not self._stop.is_set():
            started = time.perf_counter()
            config = self._state.snapshot()
            sample_rate = config.sample_rate
            n = np.arange(block, dtype=np.float32)
            t = (n0 + n) / sample_rate

            audio = 0.55 * np.sin(2 * np.pi * 1_000 * t)
            audio += 0.22 * np.sin(2 * np.pi * 230 * t)
            audio += 0.10 * np.sin(2 * np.pi * 3_100 * t)

            phase_inc = 2 * np.pi * (75_000 * audio) / sample_rate
            phase_series = phase + np.cumsum(phase_inc, dtype=np.float64)
            phase = float(math.fmod(phase_series[-1], 2 * math.pi))

            main = 0.72 * np.exp(1j * phase_series)
            marker_hi = 0.18 * np.exp(2j * np.pi * 180_000 * t)
            marker_lo = 0.12 * np.exp(-2j * np.pi * 320_000 * t)
            beacon = 0.08 * (1.0 + 0.6 * np.sin(2 * np.pi * 3 * t)) * np.exp(
                2j * np.pi * 640_000 * t
            )
            noise = 0.025 * (
                self._rng.standard_normal(block) + 1j * self._rng.standard_normal(block)
            )

            self._ring.push((main + marker_hi + marker_lo + beacon + noise).astype(np.complex64))
            n0 += block

            elapsed = time.perf_counter() - started
            sleep_for = max(0.0, (block / sample_rate) - elapsed)
            self._stop.wait(sleep_for)


class RtlSdrCaptureSource:
    name = "rtl-sdr"

    def __init__(self, ring: IQRingBuffer, state: RadioState, settings: Settings):
        self._ring = ring
        self._state = state
        self._settings = settings
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._sdr = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="rtl-sdr-capture", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._sdr is not None and hasattr(self._sdr, "cancel_read_async"):
            try:
                self._sdr.cancel_read_async()
            except Exception:  # pragma: no cover - hardware cleanup best effort
                LOGGER.exception("failed to cancel rtl-sdr async read")
        if self._thread:
            self._thread.join(timeout=2.0)

    def _configure(self) -> None:
        config = self._state.snapshot()
        self._sdr.sample_rate = config.sample_rate
        self._sdr.center_freq = config.center_freq
        self._sdr.freq_correction = config.ppm
        if config.gain == "auto" or config.gain == "-1":
            self._sdr.gain = "auto"
        else:
            self._sdr.gain = float(config.gain)

    def _run(self) -> None:
        from rtlsdr import RtlSdr

        self._sdr = RtlSdr()
        self._configure()
        last_config = self._state.snapshot()

        def callback(samples, _sdr) -> None:
            nonlocal last_config
            if self._stop.is_set():
                raise StopIteration

            config = self._state.snapshot()
            if config != last_config:
                self._configure()
                last_config = config

            self._ring.push(np.asarray(samples, dtype=np.complex64))

        try:
            self._sdr.read_samples_async(callback, self._settings.block_size)
        except StopIteration:
            pass
        finally:
            self._sdr.close()
            self._sdr = None


def build_capture_source(
    ring: IQRingBuffer,
    state: RadioState,
    settings: Settings,
) -> CaptureSource:
    if settings.source == "sim":
        return SimulatedCaptureSource(ring, state, settings)

    if settings.source in {"rtl", "rtlsdr", "rtl-sdr"}:
        return RtlSdrCaptureSource(ring, state, settings)

    try:
        import rtlsdr  # noqa: F401
    except ImportError:
        LOGGER.warning("pyrtlsdr is not installed; using simulated SDR source")
        return SimulatedCaptureSource(ring, state, settings)

    return RtlSdrCaptureSource(ring, state, settings)

