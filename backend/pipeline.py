from __future__ import annotations

import asyncio
import contextlib
import logging
import time

import numpy as np

from .capture import CaptureSource, IQRingBuffer, build_capture_source
from .config import Settings
from .dsp import compute_waterfall, demodulate
from .protocol import make_waterfall_frame
from .state import RadioConfig, RadioState


LOGGER = logging.getLogger(__name__)


class BroadcastHub:
    def __init__(self, max_queue: int = 2):
        self._queues: set[asyncio.Queue[bytes]] = set()
        self._max_queue = max_queue

    @property
    def client_count(self) -> int:
        return len(self._queues)

    def subscribe(self) -> asyncio.Queue[bytes]:
        queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=self._max_queue)
        self._queues.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[bytes]) -> None:
        self._queues.discard(queue)

    def publish(self, payload: bytes) -> None:
        for queue in tuple(self._queues):
            if queue.full():
                with contextlib.suppress(asyncio.QueueEmpty):
                    queue.get_nowait()
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
        self.waterfall = BroadcastHub(max_queue=2)
        self.audio = BroadcastHub(max_queue=4)
        self.source: CaptureSource = build_capture_source(self.ring, self.state, settings)
        self._task: asyncio.Task[None] | None = None
        self._running = False
        self._seq = 0
        self._last_waterfall_at = 0.0

    @property
    def status(self) -> dict[str, object]:
        return {
            "source": self.source.name,
            "running": self._running,
            "ring_depth": self.ring.depth,
            "dropped_blocks": self.ring.dropped_blocks,
            "waterfall_clients": self.waterfall.client_count,
            "audio_clients": self.audio.client_count,
            "config": self.state.snapshot().as_dict(),
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
        min_waterfall_interval = 1.0 / max(1.0, self.settings.waterfall_fps)
        while self._running:
            iq = self.ring.pop()
            if iq is None:
                await asyncio.sleep(0.002)
                continue

            config = self.state.snapshot()
            now = time.perf_counter()

            if self.audio.client_count:
                pcm = await asyncio.to_thread(
                    demodulate,
                    iq,
                    config.mode,
                    sample_rate=config.sample_rate,
                    audio_rate=self.settings.audio_rate,
                )
                self.audio.publish(np.asarray(pcm, dtype="<f4").tobytes(order="C"))

            if self.waterfall.client_count and now - self._last_waterfall_at >= min_waterfall_interval:
                bins = await asyncio.to_thread(compute_waterfall, iq, config.fft_size)
                self.waterfall.publish(make_waterfall_frame(self._seq, config, bins))
                self._seq = (self._seq + 1) & 0xFFFFFFFF
                self._last_waterfall_at = now

