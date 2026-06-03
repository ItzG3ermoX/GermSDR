# GermSDR WebSDR

High-performance WebSDR starter for RTL-SDR v4 receivers with a Python DSP backend,
binary WebSockets, a WebGL 2 waterfall, and Web Audio playback.

The app defaults to `SDR_SOURCE=auto`: it uses an RTL-SDR when `pyrtlsdr` is
available, otherwise it starts a simulated IQ source so the browser UI can be
tested without hardware.

## Project Layout

```text
backend/
  capture.py      IQ ring buffer plus simulated and RTL-SDR capture sources
  dsp.py          FFT, WBFM, AM, SSB, and CW demodulation helpers
  pipeline.py     Single DSP pump that broadcasts waterfall and audio frames
  protocol.py     20-byte aligned binary waterfall frame format
  server.py       FastAPI REST and WebSocket endpoints
frontend/
  src/            Vanilla TypeScript WebGL/Web Audio client
  public/         AudioWorklet processor
tests/            Focused DSP and protocol tests
```

## Run Locally

Backend:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
$env:SDR_SOURCE = "sim"
.\.venv\Scripts\python -m uvicorn backend.server:app --host 127.0.0.1 --port 8080
```

Frontend:

```powershell
cd frontend
npm install
npm run dev
```

Open `http://127.0.0.1:5173`. The Vite dev server proxies `/api` and `/ws` to
the backend on port `8080`.

## RTL-SDR v4

Install the optional SDR dependency set when the dongle and driver are ready:

```powershell
.\.venv\Scripts\python -m pip install -r requirements-sdr.txt
$env:SDR_SOURCE = "rtl"
```

Useful environment variables:

```text
SDR_FREQ=100800000
SDR_RATE=2400000
SDR_GAIN=auto
SDR_PPM=0
SDR_FFT=32768
SDR_BLOCK=65536
```

## Docker

On Linux hosts with USB pass-through:

```bash
docker compose up --build
```

The compose file maps `/dev/bus/usb` and starts the backend on port `8080`.

## Protocol

Waterfall frames use a 20-byte network-endian header followed by little-endian
float32 bins:

```text
uint32 seq
float64 center_freq
float32 sample_rate
uint16 fft_size
uint16 flags
float32 bins[fft_size]
```

The explicit padding keeps the payload aligned for browser `Float32Array`
views. If a client GPU cannot allocate a 32,768-wide texture, the renderer keeps
the full-resolution websocket frame and peak-reduces it into the largest
supported single-channel texture.
