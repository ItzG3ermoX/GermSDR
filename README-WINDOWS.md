# GermSDR Windows Deployment Guide

This guide describes how to run GermSDR on Windows.

## Prerequisites
1.  **Python 3.11+**: Ensure Python is installed and added to your PATH.
2.  **Node.js 20.19+ and npm**: Required to build the frontend.
3.  **RTL-SDR Drivers**: You must install the Zadig drivers for your RTL-SDR dongle.

## Setup Instructions

### 1. Environment Setup
Open PowerShell or Command Prompt in the project folder:

```powershell
# Create virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install backend dependencies
python -m pip install -r requirements.txt

# (Optional) Install RTL-SDR specific dependencies
python -m pip install -r requirements-sdr.txt
```

### 2. Frontend Build
```powershell
cd frontend
npm install
npm run build
cd ..
```

### 3. Run the Server
For development (simulated mode):
```powershell
$env:SDR_SOURCE="sim"
.\.venv\Scripts\python -m uvicorn backend.server:app --host 127.0.0.1 --port 8080
```

For RTL-SDR dongle (after installing drivers):
```powershell
$env:SDR_SOURCE="rtl"
.\.venv\Scripts\python -m uvicorn backend.server:app --host 127.0.0.1 --port 8080
```

Open `http://127.0.0.1:8080` in your browser.
