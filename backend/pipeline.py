from __future__ import annotations

import asyncio
import contextlib
import logging
import math

import numpy as np

from .capture import CaptureSource, IQRingBuffer, build_capture_source
from .config import Settings
import dataclasses

from .dsp import (
    DemodState,
    _signal_power_db,
    compute_waterfall_zoom,
    demodulate,
)
from .protocol import make_waterfall_frame
from .state import RadioConfig, RadioState


LOGGER = logging.getLogger(__name__)


class BroadcastHub:
    """Simple pub/sub using per-client asyncio.Queue.

    ``drop_newest`` controls overflow behaviour:
      True  (waterfall) → drop the OLDEST frame when full, always deliver the latest.
      False (audio)     → skip publishing (drop newest) when full, preserving stream continuity.

    On unsubscribe we put a sentinel (None) so any task currently blocked
    on queue.get() wakes up immediately instead of hanging forever.
    Consumers must treat None as "hub closed for this subscriber".
    """

    def __init__(self, max_queue: int = 2, drop_newest: bool = False):
        self._queues: set[asyncio.Queue[bytes | None]] = set()
        self._max_queue = max_queue
        self._drop_newest = drop_newest

    @property
    def client_count(self) -> int:
        return len(self._queues)

    def subscribe(self) -> asyncio.Queue[bytes | None]:
        queue: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=self._max_queue)
        self._queues.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[bytes | None]) -> None:
        self._queues.discard(queue)
        # Wake up any waiter blocked in queue.get() so the consumer task
        # can exit its loop cleanly instead of leaking.
        try:
            queue.put_nowait(None)
        except asyncio.QueueFull:
            pass

    def publish(self, payload: bytes) -> None:
        for queue in tuple(self._queues):
            if queue.full():
                if self._drop_newest:
                    # Drop oldest so the newest frame (always the latest
                    # waterfall row) gets through without delay.
                    with contextlib.suppress(asyncio.QueueEmpty):
                        queue.get_nowait()
                    with contextlib.suppress(asyncio.QueueFull):
                        queue.put_nowait(payload)
                else:
                    # Audio: dropping oldest creates audible gaps. Skip
                    # this packet entirely — the player's jitter buffer
                    # carries on with what it has, continuous but slightly
                    # more delayed. No gap, no click.
                    pass
            else:
                with contextlib.suppress(asyncio.QueueFull):
                    queue.put_nowait(payload)


class SdrPipeline:
    def __init__(self, settings: Settings):
        initial_config = RadioConfig(
            center_freq=settings.center_freq,
            sample_rate=settings.sample_rate,
            gain=settings.gain,
            ppm=settings.ppm,
            fft_size=settings.fft_size,
        )
        self.settings = settings
        self.state = RadioState(initial_config)
        self.ring = IQRingBuffer(settings.ring_slots, settings.block_size)
        self.waterfall = BroadcastHub(max_queue=2, drop_newest=True)
        self.audio = BroadcastHub(max_queue=4)
        self.demod_state = DemodState()
        self.peak_hold: np.ndarray | None = None
        self.source: CaptureSource = build_capture_source(self.ring, self.state, settings)
        self._task: asyncio.Task[None] | None = None
        self._running = False
        self._seq = 0
        self._last_waterfall_at = 0.0
        # Signal strength tracking for status / squelch
        self._signal_strength_db: float = -160.0
        self._squelch_open: bool = True

    @property
    def signal_strength_db(self) -> float:
        return self._signal_strength_db

    @property
    def status(self) -> dict[str, object]:
        device_info = {}
        if hasattr(self.source, "device_info"):
            device_info = self.source.device_info
        return {
            "source": self.source.name,
            "running": self._running and self.source.healthy,
            "ring_depth": self.ring.depth,
            "dropped_blocks": self.ring.dropped_blocks,
            "waterfall_clients": self.waterfall.client_count,
            "audio_clients": self.audio.client_count,
            "config": self.state.snapshot().as_dict(),
            "tuned_freq": self.state.tuned_freq,
            "tune_offset": self.state.tune_offset,
            "signal_strength_db": self._signal_strength_db,
            "squelch_open": self._squelch_open,
            "device_info": device_info,
        }

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self.source.start()
        self._task = asyncio.create_task(self._pump(), name="sdr-dsp-pump")
        LOGGER.info("SDR pipeline started with %s source", self.source.name)

    async def stop(self) -> None:
        self._running = False
        self.source.stop()
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    async def _pump(self) -> None:
        # Target waterfall ROW rate (rows/sec). We hold this CONSTANT regardless of
        # the hardware sample rate. One capture block spans block_size/rate seconds;
        # at 2.4 MHz that's ~27 ms (fine), but a hardware-rate auto-switch down to
        # 0.25 MHz makes one block ~262 ms -> only ~4 rows/sec -> the waterfall
        # visibly stutters at deep zoom. So we slice each block into as many time
        # sub-windows as needed to keep ~target_row_rate rows/sec at every rate.
        target_row_rate = max(1.0, float(self.settings.waterfall_fps))
        last_pump_rate: int = 0
        last_pump_center: int = 0
        last_pump_offset: float = 0.0
        while self._running:
            iq = self.ring.pop()
            if iq is None:
                await asyncio.sleep(0.002)
                continue

            config = self.state.snapshot()
            offset = self.state.tune_offset

            # Stale-block guard: skip one IQ block whenever any parameter that
            # affects demodulation changed between pump iterations. The previous
            # block was captured at the OLD setting, so frequency-shifting /
            # channelizing / demodulating it at the NEW setting produces garbage
            # audio for one block (~27-262 ms).
            #
            # Guards:
            #   sample_rate  → phase increment and decimation factors change
            #   center_freq  → signal of interest moved (hardware retune)
            #   tune_offset  → click-to-tune moved the listening point within
            #                   the band (every waterfall click!)
            #
            # The capture thread calls ring.clear() on hardware changes, but
            # this catches the race where clear() runs after pop().
            stale = False
            if last_pump_rate != 0:
                if config.sample_rate != last_pump_rate:
                    stale = True
                elif config.center_freq != last_pump_center:
                    stale = True
                elif offset != last_pump_offset:
                    stale = True
            if stale:
                # A state change can race the capture thread's reconfiguration
                # check. Drain every queued block rather than assuming only the
                # one we popped was captured with the previous parameters.
                self.ring.clear()
                last_pump_rate = config.sample_rate
                last_pump_center = config.center_freq
                last_pump_offset = offset
                continue
            last_pump_rate = config.sample_rate
            last_pump_center = config.center_freq
            last_pump_offset = offset

            if self.audio.client_count:
                pcm = await asyncio.to_thread(
                    demodulate,
                    iq,
                    config.mode,
                    sample_rate=config.sample_rate,
                    audio_rate=self.settings.audio_rate,
                    # Listen at the clicked offset within the band (0 = centre).
                    freq_offset_hz=self.state.tune_offset,
                    bandwidth_hz=config.demod_bw,
                    squelch_threshold=config.squelch,
                    state=self.demod_state,
                )
                self.audio.publish(np.asarray(pcm, dtype="<f4").tobytes(order="C"))

                # Update signal strength and squelch status from demod state.
                # When audio is on, power is measured from the channelized IF
                # (zoom-independent, reflects only the tuned signal).
                self._signal_strength_db = self.demod_state.last_signal_power_db
                self._squelch_open = self.demod_state.squelch_open
            elif self.waterfall.client_count:
                # No active audio path: measure signal power from the raw
                # capture block. This includes the full captured band, so it
                # will read higher than the channelized IF measurement used
                # when audio is on — the signal meter may shift a few dB when
                # toggling audio.
                self._signal_strength_db = _signal_power_db(iq)
            else:
                self._signal_strength_db = -160.0

            if self.waterfall.client_count and iq.size:
                zoom, pan = self.state.view

                # How many rows this block should yield to hold the target rate.
                # Round UP (ceil) so a block spanning >1 row period is never under-
                # sampled into a single row (e.g. a 55 ms block at 1.2 MHz would
                # round to 1 -> only ~18 rows/s and slightly chunky; ceil -> 2).
                block_seconds = iq.size / max(1.0, float(config.sample_rate))
                rows = math.ceil(block_seconds * target_row_rate - 1e-6)
                rows = max(1, min(rows, 16))  # never explode work on a slow block
                win = iq.size // rows

                # PACING: a slow (low-rate / deep-zoom) block represents a long
                # span of real time but we compute all its sub-rows in a few ms.
                # Publishing them back-to-back makes the waterfall lurch forward
                # several rows then freeze until the next block (~262 ms at
                # 0.25 MHz) -- which reads as "rows come slowly". Spread the rows
                # across (slightly less than) the block's real duration so they
                # stream out smoothly at the target rate. Pace to 90% so we always
                # finish before the next block is ready (never fall behind).
                row_interval = (block_seconds * 0.9 / rows) if rows > 1 else 0.0

                for r in range(rows):
                    if r > 0 and row_interval > 0.0:
                        await asyncio.sleep(row_interval)
                    lo = r * win
                    hi = iq.size if r == rows - 1 else lo + win
                    chunk = iq[lo:hi]

                    bins, slice_center, slice_rate = await asyncio.to_thread(
                        compute_waterfall_zoom,
                        chunk,
                        config.fft_size,
                        config.sample_rate,
                        config.center_freq,
                        zoom=zoom,
                        pan=pan,
                    )

                    # Peak-hold update
                    if self.peak_hold is None or self.peak_hold.size != bins.size:
                        self.peak_hold = bins.copy()
                    else:
                        self.peak_hold = np.maximum(self.peak_hold, bins)

                    # Header carries the *slice* centre/rate so the client's freq
                    # math (ruler, click-to-tune) describes exactly what's on screen.
                    frame_config = dataclasses.replace(
                        config,
                        center_freq=int(round(slice_center)),
                        sample_rate=int(round(slice_rate)),
                    )
                    self.waterfall.publish(make_waterfall_frame(self._seq, frame_config, bins))
                    self._seq = (self._seq + 1) & 0xFFFFFFFF
