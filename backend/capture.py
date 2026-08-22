from __future__ import annotations

import logging
import math
import os
import sys
import threading
import time
import ctypes
from typing import Protocol

# ---------------------------------------------------------------------------
# WINDOWS DLL COMPATIBILITY & V4 PATCH
#
# Problem: pyrtlsdr calls ctypes.CDLL("librtlsdr") which on Windows opens a
# *second* copy of the DLL with its own internal state. The device handle
# (dev_p) returned by the first copy's rtlsdr_open is then passed into the
# second copy's rtlsdr_read_async — which sees an uninitialised struct and
# crashes at a fixed offset (0x24 = 36) from a null base pointer.
#
# Fix strategy:
#  1. Pre-load the DLL using the FULL PATH so Windows returns the SAME handle
#  2. Patch CDLL.__getattr__ to return no-op stubs for missing symbols
#  3. Use read_sync() instead of read_async() for V4 driver compatibility
# ---------------------------------------------------------------------------
if sys.platform == "win32":
    import ctypes.util as _ctypes_util
    _orig_find_library = _ctypes_util.find_library

    # Locate librtlsdr.dll relative to the project root (the directory that
    # contains the `backend` package), overridable via the SDR_DLL_DIR env var
    # for non-standard installs. No more hardcoded user paths.
    from pathlib import Path as _Path

    _dll_dir = os.environ.get("SDR_DLL_DIR") or str(_Path(__file__).resolve().parents[1])
    _rtlsdr_dll_path = os.path.join(_dll_dir, "librtlsdr.dll")

    def _patched_find_library(name: str):
        if name in ("rtlsdr", "librtlsdr", "librtlsdr.dll", "rtlsdr.dll"):
            if os.path.exists(_rtlsdr_dll_path):
                return _rtlsdr_dll_path
        return _orig_find_library(name)

    _ctypes_util.find_library = _patched_find_library

    class _DummyCtypesFunc:
        def __init__(self, name: str):
            self.name = name
            self.argtypes = None
            self.restype = None
            self.errcheck = None
            self._warned = False

        def __call__(self, *args, **kwargs):
            if not self._warned:
                print(f"WARNING: {self.name} is not available in the loaded librtlsdr DLL "
                      "(calls will silently return 0)")
                self._warned = True
            return 0

    _orig_cdll_getattr = ctypes.CDLL.__getattr__

    def _patched_cdll_getattr(self, name: str):
        try:
            return _orig_cdll_getattr(self, name)
        except AttributeError:
            if name.startswith("rtlsdr_"):
                return _DummyCtypesFunc(name)
            raise

    ctypes.CDLL.__getattr__ = _patched_cdll_getattr

    try:
        for dep in ("msvcr100.dll", "pthreadVC2.dll"):
            p = os.path.join(_dll_dir, dep)
            if os.path.exists(p):
                ctypes.WinDLL(p)

        # Use LoadLibraryEx to force consistent loading
        kernel32 = ctypes.windll.kernel32
        LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR = 0x00000100
        LOAD_LIBRARY_SEARCH_DEFAULT = 0x00001000
        _h = kernel32.LoadLibraryExW(
            _rtlsdr_dll_path,
            None,
            LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT,
        )
        if _h:
            print(f"SUCCESS: RTL-SDR V4 driver loaded (handle={_h})")
        else:
            raise OSError("LoadLibraryExW returned NULL")
    except Exception as _e:
        print(f"WARNING: Pre-load failed: {_e}")
        print("Will rely on pyrtlsdr to load the DLL...")

import numpy as np

from .config import Settings
from .state import RadioState

LOGGER = logging.getLogger(__name__)

# Capture recovery constants
_MAX_RETRY_DELAY = 30.0
_RETRY_BACKOFF = 2.0


class IQRingBuffer:
    """Bounded single-producer/single-consumer IQ ring."""

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

    def clear(self) -> None:
        """Drain all blocks from the ring under lock.

        Call on sample-rate change so the pump never processes a block captured
        at the old rate at the new rate (which phase-rotates at the wrong speed
        and decimates by the wrong factor, producing a block of garbage audio).
        """
        with self._lock:
            self._read_seq = self._write_seq

    def pop(self) -> np.ndarray | None:
        with self._lock:
            if self._read_seq == self._write_seq:
                return None
            out = self._buffer[self._read_seq % self.slots].copy()
            self._read_seq += 1
            return out


class CaptureSource(Protocol):
    name: str

    @property
    def healthy(self) -> bool:
        ...

    def start(self) -> None:
        ...

    def stop(self) -> None:
        ...


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
        self._thread = threading.Thread(
            target=self._run, name="sdr-sim-capture", daemon=True
        )
        self._thread.start()

    @property
    def healthy(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

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
                self._rng.standard_normal(block)
                + 1j * self._rng.standard_normal(block)
            )

            self._ring.push(
                (main + marker_hi + marker_lo + beacon + noise).astype(np.complex64)
            )
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
        self._lib = None
        self._retry_delay: float = 1.0
        self._recovery_mode: bool = False
        self._healthy: bool = False

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._retry_delay = 1.0
        self._recovery_mode = False
        self._healthy = False
        self._thread = threading.Thread(
            target=self._run, name="rtl-sdr-capture", daemon=True
        )
        self._thread.start()

    @property
    def healthy(self) -> bool:
        return self._healthy

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def _try_reinit(self) -> bool:
        """Try to re-open the SDR device after a failure. Returns True on success."""
        from rtlsdr import RtlSdr
        try:
            LOGGER.info("Attempting SDR device re-init (%.1fs delay)...", self._retry_delay)
            self._stop.wait(self._retry_delay)
            if self._stop.is_set():
                return False
            self._sdr = RtlSdr()
            if not self._sdr.dev_p:
                self._sdr.close()
                self._sdr = None
                raise RuntimeError("re-init: dev_p is null")
            try:
                import rtlsdr.librtlsdr as librtlsdr
                self._lib = librtlsdr
            except Exception:
                self._lib = None
            self._configure()
            LOGGER.info("SDR device re-init succeeded")
            self._retry_delay = 1.0  # Reset backoff on success
            self._recovery_mode = False
            return True
        except Exception as exc:
            LOGGER.warning("SDR re-init failed: %s, retrying...", exc)
            self._retry_delay = min(self._retry_delay * _RETRY_BACKOFF, _MAX_RETRY_DELAY)
            return False

    def _apply_gain(self, config) -> None:
        """Apply gain settings - skip if auto-gain fails (V4 driver quirk)."""
        if not self._sdr or not self._lib:
            return

        try:
            if config.gain in ("auto", "-1"):
                # Try auto-gain, but don't crash if it fails
                LOGGER.debug("Setting tuner gain mode: auto (AGC)")
                r = self._lib.rtlsdr_set_tuner_gain_mode(self._sdr.dev_p, 0)
                if r != 0:
                    LOGGER.warning(
                        "Auto-gain returned %d (V4 driver), falling back to "
                        "30 dB manual gain so the receiver isn't deaf", r
                    )
                    # V4 R828D tuner often rejects the auto-gain API call.
                    # Without a fallback the tuner stays at its power-on default
                    # (sometimes minimum gain -> flat noise floor + 'dead' receiver).
                    # Set manual mode + a sensible middle gain as a safe default.
                    self._lib.rtlsdr_set_tuner_gain_mode(self._sdr.dev_p, 1)
                    self._lib.rtlsdr_set_tuner_gain(self._sdr.dev_p, 300)  # 30 dB
            else:
                gain_db = float(config.gain)
                gain_tenths = int(gain_db * 10)
                LOGGER.debug("Setting tuner gain: %.1f dB (%d tenths)", gain_db, gain_tenths)
                r1 = self._lib.rtlsdr_set_tuner_gain_mode(self._sdr.dev_p, 1)
                r2 = self._lib.rtlsdr_set_tuner_gain(self._sdr.dev_p, gain_tenths)
                if r1 != 0 or r2 != 0:
                    LOGGER.warning(
                        "Manual gain returned mode=%d gain=%d, ignoring (V4 driver)", r1, r2
                    )
        except Exception as exc:
            LOGGER.warning("Gain setting failed: %s", exc)

    def _apply_rtl_agc(self, config) -> None:
        """Enable/disable RTL-SDR digital AGC."""
        if not self._sdr or not self._lib:
            return
        try:
            enabled = 1 if config.rtl_agc else 0
            r = self._lib.rtlsdr_set_agc_mode(self._sdr.dev_p, enabled)
            if r != 0:
                LOGGER.warning("RTL AGC returned %d (V4 driver may not support this)", r)
            else:
                LOGGER.debug("RTL AGC set to %d", enabled)
        except Exception as exc:
            LOGGER.warning("RTL AGC failed: %s", exc)

    def _apply_tuner_bandwidth(self, sample_rate: int) -> None:
        """Narrow the R828D tuner IF filter to prevent ADC overload.

        At 2.4 MHz sample rate, librtlsdr sets the R828D internal IF filter to
        ~2.5-3.0 MHz by default (it tracks the digital sample rate). This wide
        RF bandwidth lets 2× the noise power into the 8-bit ADC, reducing its
        effective dynamic range via clipping/intermodulation within the
        300 kHz IF passband used for WBFM.

        We cap it to 1.5 MHz so the waterfall still shows ~2/3 of the capture
        band with real signal (edges get quieter, not dead), while cutting the
        RF noise floor by ~40%. At lower sample rates the filter matches or
        slightly exceeds Nyquist so there is no penalty.
        """
        if not self._sdr or not self._lib:
            return
        try:
            func = getattr(self._lib, "rtlsdr_set_tuner_bandwidth", None)
            if func is None:
                LOGGER.debug("rtlsdr_set_tuner_bandwidth not available")
                return

            # At sample rates above 1.5 MHz, cap the tuner IF to 1.5 MHz so the
            # 8-bit ADC doesn't get swamped by excess RF noise. At lower rates,
            # match the sample rate (or floor 600 kHz for the 250k ladder step).
            bw = max(600_000, min(sample_rate, 1_500_000))
            r = func(self._sdr.dev_p, bw)
            if r != 0:
                LOGGER.warning("rtlsdr_set_tuner_bandwidth(%d) returned %d", bw, r)
            else:
                LOGGER.info("R828D IF bandwidth set to %d Hz (sample_rate=%d)", bw, sample_rate)
        except Exception as exc:
            LOGGER.warning("Tuner bandwidth setting failed: %s", exc)

    def _apply_bias_tee(self, config) -> None:
        """Enable/disable Bias-T (if supported by the dongle)."""
        if not self._sdr or not self._lib:
            return
        try:
            # rtlsdr_set_bias_tee may not exist in all driver versions
            if hasattr(self._lib, "rtlsdr_set_bias_tee"):
                enabled = 1 if config.bias_tee else 0
                r = self._lib.rtlsdr_set_bias_tee(self._sdr.dev_p, enabled)
                if r != 0:
                    LOGGER.warning("Bias-T returned %d", r)
                else:
                    LOGGER.debug("Bias-T set to %d", enabled)
            else:
                LOGGER.debug("rtlsdr_set_bias_tee not available in this driver")
        except Exception as exc:
            LOGGER.warning("Bias-T failed: %s", exc)

    def _apply_direct_sampling(self, config) -> None:
        """Set direct sampling mode (for HF reception).
        0 = off, 1 = I-channel, 2 = Q-channel."""
        if not self._sdr or not self._lib:
            return
        try:
            mode = int(config.direct_sampling)
            if hasattr(self._lib, "rtlsdr_set_direct_sampling"):
                r = self._lib.rtlsdr_set_direct_sampling(self._sdr.dev_p, mode)
                if r != 0:
                    LOGGER.warning("Direct sampling returned %d", r)
                else:
                    LOGGER.debug("Direct sampling set to %d", mode)
            else:
                LOGGER.debug("rtlsdr_set_direct_sampling not available")
        except Exception as exc:
            LOGGER.warning("Direct sampling failed: %s", exc)

    @property
    def device_info(self) -> dict[str, object]:
        """Return RTL-SDR device info, or empty dict if not available."""
        if not self._sdr or not hasattr(self._sdr, "get_tuner_type"):
            return {}
        try:
            info: dict[str, object] = {}
            if hasattr(self._sdr, "get_device_usb_strings"):
                man, prod, serial = self._sdr.get_device_usb_strings()
                info["manufacturer"] = man
                info["product"] = prod
                info["serial"] = serial
            if hasattr(self._sdr, "get_tuner_type"):
                info["tuner"] = str(self._sdr.get_tuner_type())
            if hasattr(self._sdr, "get_xtal_freq"):
                info["xtal_freq"] = self._sdr.get_xtal_freq()
            if self._lib and hasattr(self._lib, "rtlsdr_get_tuner_gains"):
                try:
                    gains_arr = (ctypes.c_int * 100)()
                    count = self._lib.rtlsdr_get_tuner_gains(self._sdr.dev_p, gains_arr, 100)
                    if count > 0:
                        info["available_gains"] = [g / 10.0 for g in gains_arr[:count]]
                except Exception:
                    pass
            return info
        except Exception as exc:
            LOGGER.warning("Device info query failed: %s", exc)
            return {}

    def _configure(self) -> None:
        """Push current RadioState to the hardware."""
        config = self._state.snapshot()

        self._sdr.sample_rate = config.sample_rate
        self._sdr.center_freq = config.center_freq

        # Explicit tuner IF bandwidth: MUST come AFTER center_freq because
        # librtlsdr internally resets the tuner bandwidth EVERY time you set
        # center_freq (r82xx_set_freq → r82xx_set_receiver → r82xx_set_bandwidth).
        # Without this explicit override, 2.4 MHz gets a ~2.5 MHz IF filter
        # that swamps the 8-bit ADC with RF noise, making the channelized
        # audio sound grainy.
        self._apply_tuner_bandwidth(config.sample_rate)

        # V4 driver rejects ppm != 0 - ignore errors
        try:
            if config.ppm != 0:
                self._sdr.freq_correction = int(config.ppm)
        except Exception as exc:
            LOGGER.warning("Frequency correction: %s", exc)

        self._apply_gain(config)
        self._apply_rtl_agc(config)
        self._apply_bias_tee(config)
        self._apply_direct_sampling(config)

    def _run(self) -> None:
        from rtlsdr import RtlSdr

        LOGGER.info("Opening RTL-SDR device...")
        self._sdr = RtlSdr()

        # Get the lib handle to check for missing symbols
        try:
            import rtlsdr.librtlsdr as librtlsdr

            self._lib = librtlsdr
            LOGGER.info(
                "librtlsdr handle: id=%s", id(librtlsdr)
            )
        except Exception as exc:
            LOGGER.warning("Could not get librtlsdr handle: %s", exc)
            self._lib = None

        if not self._sdr.dev_p:
            self._sdr.close()
            raise RuntimeError(
                "RTL-SDR opened but dev_p is null. "
                "Likely cause: DLL loaded twice (split internal state) or "
                "device already open in another process."
            )

        self._configure()
        self._healthy = True
        last_config = self._state.snapshot()

        # Use read_sync for V4 driver compatibility
        block_size = self._settings.block_size

        LOGGER.info(
            "Starting capture: sample_rate=%d center_freq=%d block_size=%d",
            last_config.sample_rate,
            last_config.center_freq,
            block_size,
        )

        try:
            while not self._stop.is_set():
                # Check for config changes
                config = self._state.snapshot()
                if config != last_config:
                    self._configure()
                    self._ring.clear()
                    last_config = config

                # Read samples synchronously (works with V4 driver)
                try:
                    samples = self._sdr.read_samples(block_size)
                    self._ring.push(np.asarray(samples, dtype=np.complex64))
                    self._recovery_mode = False
                except Exception as exc:
                    LOGGER.error("read_samples failed: %s", exc)
                    # Enter retry loop instead of permanent exit
                    while not self._stop.is_set():
                        if self._try_reinit():
                            break
                        # _try_reinit already waited and backed off
                    continue  # Resume capture loop after retry

                if self._stop.is_set():
                    break

        except Exception as exc:
            LOGGER.exception("Capture loop error: %s", exc)
        finally:
            self._healthy = False
            try:
                if self._sdr:
                    self._sdr.close()
            except Exception as exc:
                LOGGER.warning("Error closing RTL-SDR: %s", exc)
            self._sdr = None
            LOGGER.info("RTL-SDR device closed.")


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
        from rtlsdr import RtlSdr

        device = RtlSdr()
        device.close()
    except ImportError:
        LOGGER.warning("pyrtlsdr is not installed; using simulated SDR source")
        return SimulatedCaptureSource(ring, state, settings)
    except Exception as exc:
        # Auto mode must remain usable when a driver is installed but no usable
        # dongle is attached (or it is already claimed by another process).
        LOGGER.warning("RTL-SDR unavailable (%s); using simulated SDR source", exc)
        return SimulatedCaptureSource(ring, state, settings)

    try:
        if RtlSdr.get_device_count() < 1:
            LOGGER.warning("No RTL-SDR device found; using simulated SDR source")
            return SimulatedCaptureSource(ring, state, settings)
    except (OSError, RuntimeError) as exc:
        LOGGER.warning("RTL-SDR discovery failed (%s); using simulated SDR source", exc)
        return SimulatedCaptureSource(ring, state, settings)

    return RtlSdrCaptureSource(ring, state, settings)