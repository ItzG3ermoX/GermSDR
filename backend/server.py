from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import Settings
from .pipeline import SdrPipeline


class TuneRequest(BaseModel):
    freq: int | None = Field(default=None, ge=1, description="Center frequency in Hz")
    mode: str | None = Field(default=None, description="wbfm, am, usb, lsb, or cw")
    gain: str | float | None = Field(default=None, description="'auto' or gain in dB")
    ppm: float | None = Field(default=None, description="Frequency correction in PPM")
    fft_size: int | None = Field(default=None, description="Requested waterfall FFT size")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    app = FastAPI(title="GermSDR WebSDR", version="1.0.0")
    pipeline = SdrPipeline(settings)
    app.state.pipeline = pipeline

    @app.on_event("startup")
    async def startup() -> None:
        await pipeline.start()

    @app.on_event("shutdown")
    async def shutdown() -> None:
        await pipeline.stop()

    @app.get("/api/status")
    async def status() -> dict[str, object]:
        return pipeline.status

    @app.post("/api/tune")
    async def tune(req: TuneRequest) -> dict[str, object]:
        try:
            config = pipeline.state.update(
                center_freq=req.freq,
                mode=req.mode,
                gain=req.gain,
                ppm=req.ppm,
                fft_size=req.fft_size,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"ok": True, "config": config.as_dict()}

    @app.websocket("/ws/waterfall")
    async def ws_waterfall(ws: WebSocket) -> None:
        await ws.accept()
        queue = pipeline.waterfall.subscribe()
        try:
            while True:
                await ws.send_bytes(await queue.get())
        except WebSocketDisconnect:
            pass
        finally:
            pipeline.waterfall.unsubscribe(queue)

    @app.websocket("/ws/audio")
    async def ws_audio(ws: WebSocket) -> None:
        await ws.accept()
        queue = pipeline.audio.subscribe()
        try:
            while True:
                await ws.send_bytes(await queue.get())
        except WebSocketDisconnect:
            pass
        finally:
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

