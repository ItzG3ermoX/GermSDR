from __future__ import annotations

import asyncio
import contextlib
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import Settings
from .pipeline import SdrPipeline
from .state import RadioState


class TuneRequest(BaseModel):
    freq: int | None = Field(
        default=None,
        ge=RadioState.TUNE_MIN_HZ,
        le=RadioState.TUNE_MAX_HZ,
        description="Center frequency in Hz",
    )
    mode: str | None = Field(default=None, description="wbfm, am, usb, lsb, or cw")
    gain: str | float | None = Field(default=None, description="'auto' or gain in dB")
    ppm: float | None = Field(default=None, description="Frequency correction in PPM")
    fft_size: int | None = Field(default=None, description="Requested waterfall FFT size")
    demod_bw: int | None = Field(default=None, ge=50, le=250_000, description="Demodulation bandwidth in Hz")
    squelch: float | None = Field(default=None, description="Squelch threshold in dBFS (-160 = off)")
    rtl_agc: bool | None = Field(default=None, description="RTL-SDR hardware AGC")
    bias_tee: bool | None = Field(default=None, description="Bias-T enable")
    direct_sampling: int | None = Field(default=None, ge=0, le=2, description="Direct sampling mode: 0=off, 1=I, 2=Q")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    pipeline = SdrPipeline(settings)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        await pipeline.start()
        try:
            yield
        finally:
            await pipeline.stop()

    app = FastAPI(title="GermSDR WebSDR", version="1.0.0", lifespan=lifespan)
    app.state.pipeline = pipeline

    @app.get("/api/status")
    async def status() -> dict[str, object]:
        return pipeline.status

    @app.post("/api/tune")
    async def tune(req: TuneRequest) -> dict[str, object]:
        try:
            current = pipeline.state.snapshot()
            recentered = False
            # Click-to-tune: if the requested frequency falls inside the
            # captured band we just move the listening point (tune offset) and
            # leave the hardware centre -- and therefore the waterfall -- exactly
            # where it is. Only when the target is outside the band (or close
            # enough to the edge that the demod passband would clip) do we retune
            # the hardware and recentre. "Recenter only when needed."
            if req.freq is not None:
                offset = float(req.freq) - float(current.center_freq)
                # Keep a guard band so a signal near the very edge still
                # demodulates cleanly; beyond that, recentre the hardware.
                edge = 0.45 * float(current.sample_rate)
                if abs(offset) <= edge:
                    pipeline.state.set_tune_offset(offset)
                    config = pipeline.state.update(
                        center_freq=None,
                        mode=req.mode,
                        gain=req.gain,
                        ppm=req.ppm,
                        fft_size=req.fft_size,
                        demod_bw=req.demod_bw,
                        squelch=req.squelch,
                        rtl_agc=req.rtl_agc,
                        bias_tee=req.bias_tee,
                        direct_sampling=req.direct_sampling,
                    )
                else:
                    # Out of band: retune hardware (resets offset to 0).
                    config = pipeline.state.update(
                        center_freq=int(req.freq),
                        mode=req.mode,
                        gain=req.gain,
                        ppm=req.ppm,
                        fft_size=req.fft_size,
                        demod_bw=req.demod_bw,
                        squelch=req.squelch,
                        rtl_agc=req.rtl_agc,
                        bias_tee=req.bias_tee,
                        direct_sampling=req.direct_sampling,
                    )
                    recentered = True
            else:
                config = pipeline.state.update(
                    center_freq=None,
                    mode=req.mode,
                    gain=req.gain,
                    ppm=req.ppm,
                    fft_size=req.fft_size,
                    demod_bw=req.demod_bw,
                    squelch=req.squelch,
                    rtl_agc=req.rtl_agc,
                    bias_tee=req.bias_tee,
                    direct_sampling=req.direct_sampling,
                )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "ok": True,
            "config": config.as_dict(),
            "tuned_freq": pipeline.state.tuned_freq,
            "tune_offset": pipeline.state.tune_offset,
            "recentered": recentered,
        }

    @app.websocket("/ws/waterfall")
    async def ws_waterfall(ws: WebSocket) -> None:
        await ws.accept()
        queue = pipeline.waterfall.subscribe()

        async def receive_view() -> None:
            # Client sends {"zoom": float, "pan": float, "pan_hz": float} as JSON
            # text to drive server-side zoom/pan. pan_hz (absolute view-centre
            # frequency) comes from drag-to-pan. Last-writer-wins; no ack needed.
            try:
                while True:
                    msg = await ws.receive_json()
                    pipeline.state.set_view(
                        zoom=msg.get("zoom"),
                        pan=msg.get("pan"),
                        pan_hz=msg.get("pan_hz"),
                    )
            except (WebSocketDisconnect, KeyError, ValueError, TypeError):
                pass

        receiver = asyncio.create_task(receive_view())
        try:
            while True:
                queue_get = asyncio.create_task(queue.get())
                done, _ = await asyncio.wait(
                    {queue_get, receiver}, return_when=asyncio.FIRST_COMPLETED
                )
                if receiver in done:
                    queue_get.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await queue_get
                    break
                data = queue_get.result()
                if data is None:
                    # Unsubscribe sentinel (client gone or hub shutting down)
                    break
                try:
                    await ws.send_bytes(data)
                except (WebSocketDisconnect, RuntimeError):
                    # Client disconnected or uvicorn already closed the socket
                    # (common when Vite proxy / page reloads / rapid reconnects).
                    break
        except WebSocketDisconnect:
            pass
        except RuntimeError:
            # Catch any late RuntimeError from the send path (ASGI send after close).
            pass
        finally:
            receiver.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await receiver
            pipeline.waterfall.unsubscribe(queue)

    @app.websocket("/ws/audio")
    async def ws_audio(ws: WebSocket) -> None:
        await ws.accept()
        queue = pipeline.audio.subscribe()

        async def wait_for_disconnect() -> None:
            try:
                while True:
                    await ws.receive()
            except WebSocketDisconnect:
                return

        disconnect = asyncio.create_task(wait_for_disconnect())
        try:
            while True:
                queue_get = asyncio.create_task(queue.get())
                done, _ = await asyncio.wait(
                    {queue_get, disconnect}, return_when=asyncio.FIRST_COMPLETED
                )
                if disconnect in done:
                    queue_get.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await queue_get
                    break
                data = queue_get.result()
                if data is None:
                    break
                try:
                    await ws.send_bytes(data)
                except (WebSocketDisconnect, RuntimeError):
                    break
        except WebSocketDisconnect:
            pass
        except RuntimeError:
            pass
        finally:
            disconnect.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await disconnect
            pipeline.audio.unsubscribe(queue)

    frontend = Path(settings.frontend_dist)
    if frontend.exists():
        app.mount("/", StaticFiles(directory=frontend, html=True), name="frontend")
    else:

        @app.get("/")
        async def index_placeholder() -> dict[str, str]:
            return {
                "message": "Frontend build not found. Run `npm install` and `npm run build` in frontend/."
            }

        @app.get("/favicon.ico")
        async def favicon_placeholder() -> FileResponse:
            raise HTTPException(status_code=404)

    return app


app = create_app()
