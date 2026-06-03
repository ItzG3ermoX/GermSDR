import './styles.css';
import { WaterfallRenderer } from './waterfall';
import { connectAudio, connectWaterfall, AudioClient } from './ws-client';
import {
  formatFrequency,
  gainValue,
  getUi,
  refreshIcons,
  selectedMode,
  selectedPalette,
  setMode,
} from './ui';

interface ApiStatus {
  source: string;
  running: boolean;
  ring_depth: number;
  dropped_blocks: number;
  waterfall_clients: number;
  audio_clients: number;
  config: {
    center_freq: number;
    sample_rate: number;
    gain: string;
    mode: string;
    ppm: number;
    fft_size: number;
  };
}

const ui = getUi();
let renderer: WaterfallRenderer | undefined;
let currentFft = Number(ui.fftSelect.value);
let audio: AudioClient | undefined;
let frameCount = 0;
let lastFpsAt = performance.now();
let lastSeq = 0;

function ensureRenderer(fftSize: number): WaterfallRenderer {
  if (!renderer || fftSize !== currentFft) {
    renderer?.destroy();
    currentFft = fftSize;
    renderer = new WaterfallRenderer(ui.canvas, fftSize);
    renderer.setColorMap(selectedPalette(ui.paletteSelect));
  }
  return renderer;
}

async function tune(): Promise<void> {
  const payload = {
    freq: Number(ui.frequencyInput.value),
    mode: selectedMode(ui.modeSegments),
    gain: gainValue(ui.gainSlider),
    fft_size: Number(ui.fftSelect.value),
  };

  const response = await fetch('/api/tune', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    const message = await response.text();
    throw new Error(message);
  }

  ui.freqReadout.textContent = formatFrequency(payload.freq);
  ui.modeReadout.textContent = payload.mode.toUpperCase();
  ui.fftReadout.textContent = `${payload.fft_size} FFT`;
}

async function refreshStatus(): Promise<void> {
  const response = await fetch('/api/status');
  if (!response.ok) {
    return;
  }
  const status = (await response.json()) as ApiStatus;
  ui.sourceLabel.textContent = status.running ? status.source : 'stopped';
  ui.clientReadout.textContent = `${status.waterfall_clients + status.audio_clients} clients`;
  ui.dropReadout.textContent = `${status.dropped_blocks} drops`;
  ui.frequencyInput.value = String(status.config.center_freq);
  ui.freqReadout.textContent = formatFrequency(status.config.center_freq);
  ui.modeReadout.textContent = status.config.mode.toUpperCase();
  ui.fftReadout.textContent = `${status.config.fft_size} FFT`;
  ui.fftSelect.value = String(status.config.fft_size);
  setMode(ui.modeSegments, status.config.mode);
  ui.gainSlider.value = status.config.gain === 'auto' ? '-1' : status.config.gain;
  ui.gainOutput.textContent = status.config.gain === '-1' ? 'auto' : status.config.gain;
}

function updateRangeOutputs(): void {
  ui.gainOutput.textContent = gainValue(ui.gainSlider);
  ui.floorOutput.textContent = ui.floorSlider.value;
  ui.ceilingOutput.textContent = ui.ceilingSlider.value;
}

connectWaterfall(
  (frame) => {
    lastSeq = frame.seq;
    ensureRenderer(frame.fftSize).pushRow(frame.bins);
    ui.freqReadout.textContent = formatFrequency(frame.centerFreq);
    ui.fftReadout.textContent = `${frame.fftSize} FFT`;
    frameCount += 1;
  },
  (state) => {
    ui.sourceLabel.textContent = state;
  },
);

function draw(): void {
  const floor = Number(ui.floorSlider.value);
  const ceiling = Math.max(floor + 1, Number(ui.ceilingSlider.value));
  ensureRenderer(currentFft).draw(floor, ceiling);

  const now = performance.now();
  if (now - lastFpsAt >= 1000) {
    ui.frameReadout.textContent = `${frameCount} fps / seq ${lastSeq}`;
    frameCount = 0;
    lastFpsAt = now;
  }

  requestAnimationFrame(draw);
}

ui.tuneButton.addEventListener('click', () => {
  void tune().catch((error: unknown) => {
    ui.sourceLabel.textContent = error instanceof Error ? error.message : 'tune failed';
  });
});

ui.frequencyInput.addEventListener('keydown', (event) => {
  if (event.key === 'Enter') {
    void tune();
  }
});

ui.modeSegments.addEventListener('click', (event) => {
  const target = event.target as HTMLElement;
  const button = target.closest<HTMLButtonElement>('button[data-mode]');
  if (!button) {
    return;
  }
  setMode(ui.modeSegments, button.dataset.mode ?? 'wbfm');
  void tune();
});

ui.gainSlider.addEventListener('input', updateRangeOutputs);
ui.gainSlider.addEventListener('change', () => void tune());
ui.fftSelect.addEventListener('change', () => void tune());
ui.paletteSelect.addEventListener('change', () => renderer?.setColorMap(selectedPalette(ui.paletteSelect)));
ui.floorSlider.addEventListener('input', updateRangeOutputs);
ui.ceilingSlider.addEventListener('input', updateRangeOutputs);

ui.audioButton.addEventListener('click', async () => {
  if (audio) {
    audio.close();
    audio = undefined;
    ui.audioButton.innerHTML = '<span data-lucide="volume-2"></span>Start Audio';
    refreshIcons();
    return;
  }

  audio = await connectAudio();
  await audio.ctx.resume();
  ui.audioButton.innerHTML = '<span data-lucide="volume-x"></span>Stop Audio';
  refreshIcons();
});

updateRangeOutputs();
void refreshStatus();
window.setInterval(() => void refreshStatus(), 1500);
requestAnimationFrame(draw);
