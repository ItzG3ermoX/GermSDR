FROM node:22-slim AS frontend
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm install
COPY frontend/ ./
RUN npm run build

FROM python:3.11-slim AS backend
RUN apt-get update && apt-get install -y --no-install-recommends \
    rtl-sdr \
    librtlsdr-dev \
    libusb-1.0-0-dev \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements-sdr.txt requirements.txt ./
RUN pip install --no-cache-dir -r requirements-sdr.txt
COPY backend/ ./backend/
COPY --from=frontend /app/frontend/dist ./frontend/dist
EXPOSE 8080
CMD ["uvicorn", "backend.server:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]

