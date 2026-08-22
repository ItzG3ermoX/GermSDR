import './styles.css';
import { WaterfallRenderer } from './waterfall';
import { bandsInRange, BAND_COLORS } from './bandplan';
import { connectAudio, connectWaterfall, AudioClient, WaterfallClient } from './ws-client';
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
    demod_bw: number;
    squelch: number;
    rtl_agc: boolean;
    bias_tee: boolean;
    direct_sampling: number;
  };
  tuned_freq: number;
  tune_offset: number;
  signal_strength_db: number;
  squelch_open: boolean;
}

const ui = getUi();
let renderer: WaterfallRenderer | undefined;
let currentFft = Number(ui.fftSelect.value);
let audio: AudioClient | undefined;
let audioStarting = false;
let waterfallClient: WaterfallClient | undefined;
let gainNode: GainNode | undefined;
let frameCount = 0;
let lastFpsAt = performance.now();
let lastSeq = 0;
// Live tuning context from the most recent waterfall frame, used to map a
// click position on the canvas back to an absolute frequency. This is the
// centre/rate of the on-screen SLICE (which follows the tuned signal when
// zoomed), NOT necessarily the hardware centre.
let lastCenterFreq = Number(ui.frequencyInput.value);
let lastSampleRate = 2_400_000;
let lastBins: Float32Array | null = null;

// Can we pan the current view? Only when zoomed IN (>1x). At 1x the whole
// captured band fills the screen -- there is no extra texture data to slide
// into view. The old "drag retunes hardware" behaviour at 1x was buggy:
// the local texture shift (renderer.setDragOffset) would push the image
// off-screen (black area), and after release the settling logic would
// compute an offset >1.0 from stale frame-center data, keeping the display
// black. Repeated drags could compound into wild frequency jumps (24 MHz).
function canPanView(): boolean {
  return Boolean(renderer) && lastSampleRate > 0 && (renderer!.zoom > 1);
}

// Simple signal-aware auto contrast. We want the noise floor to sit low but
// not black, and strong signals to reach near white without clipping.
function computeAutoContrast(bins: Float32Array): { floor: number; ceiling: number } {
  let minDb = Infinity;
  let maxDb = -Infinity;
  for (let i = 0; i < bins.length; i++) {
    const v = bins[i];
    if (v < minDb) minDb = v;
    if (v > maxDb) maxDb = v;
  }
  const span = maxDb - minDb;
  // Put floor a bit above the absolute noise floor (hides grass) but still show weak signals.
  // Use ~10-15% of the dynamic range as "lift".
  const lift = Math.max(5, span * 0.10);
  let floor = Math.round(minDb + lift);
  // Ceiling a few dB below the strongest peak so carriers pop without being pure white.
  let ceiling = Math.round(maxDb - 2.5);

  // Sanity clamp to the slider ranges
  floor = Math.max(-140, Math.min(floor, -20));
  ceiling = Math.max(floor + 6, Math.min(ceiling, 0));
  return { floor, ceiling };
}
// The absolute frequency we are actually listening to. With click-to-tune the
// waterfall band does NOT recentre on it -- the demodulator follows it inside
// the captured band -- so the tuned frequency can sit off-centre on screen.
let tunedFreq = Number(ui.frequencyInput.value);

// Debounced sender for server-side zoom/pan: scrolling fires many events, but we
// only want to notify the backend DSP a few times per second. Uses an accumulating
// object so a zoom scroll in the middle of a debounce that also has a pending pan
// doesn't drop the pan — both are sent together on the next flush.
let viewSendTimer: number | undefined;
interface PendingView { zoom?: number; pan?: number }
let pendingView: PendingView = {};
function sendView(zoom: number, pan?: number): void {
  pendingView.zoom = zoom;
  if (pan !== undefined) pendingView.pan = pan;
  if (viewSendTimer) return;  // timer already running, accumulates
  viewSendTimer = window.setTimeout(() => {
    waterfallClient?.setView(pendingView.zoom!, pendingView.pan);
    pendingView = {};
    viewSendTimer = undefined;
  }, 80);
}

function ensureRenderer(fftSize: number): WaterfallRenderer {
  if (!renderer || fftSize !== currentFft) {
    renderer?.destroy();
    currentFft = fftSize;
    renderer = new WaterfallRenderer(ui.canvas, fftSize);
    renderer.setColorMap(selectedPalette(ui.paletteSelect));
    renderer.onViewChange = sendView;
    // When the wheel or pinch changes zoom, mirror it onto the slider so the
    // slider and the mouse-wheel zoom are always showing the same value.
    renderer.onZoomChange = syncZoomSlider;
    syncZoomSlider(renderer.zoom);
  }
  return renderer;
}

// The zoom slider is linear over log2(zoom): position 0 -> 1x, max -> 64x. This
// matches the mouse wheel's multiplicative steps (1.25x / 0.8x), so dragging
// the slider and scrolling the wheel feel identical -- and a given physical
// distance is the same zoom ratio anywhere along the track.
const ZOOM_SLIDER_MAX = 600; // log2(64) * 100, must match index.html max
function sliderToZoom(pos: number): number {
  const frac = pos / ZOOM_SLIDER_MAX; // 0..1
  return Math.pow(2, frac * Math.log2(WaterfallRenderer.MAX_ZOOM));
}
function zoomToSlider(zoom: number): number {
  const frac = Math.log2(zoom) / Math.log2(WaterfallRenderer.MAX_ZOOM);
  return Math.round(frac * ZOOM_SLIDER_MAX);
}
// Push a zoom value onto the slider + its readout WITHOUT firing setZoom again
// (the value is already applied). Used by onZoomChange (wheel/pinch).
function syncZoomSlider(zoom: number): void {
  ui.zoomSlider.value = String(zoomToSlider(zoom));
  ui.zoomOutput.textContent = `${zoom.toFixed(1)}x`;
}

// Europe (Region 1) FM broadcast band. Channels sit on a 100 kHz raster
// (87.5, 87.6, ... 108.0 MHz), so snapping only makes sense in here.
const FM_BAND_LO = 87_500_000;
const FM_BAND_HI = 108_000_000;
const FM_RASTER_HZ = 100_000; // 0.1 MHz

// Snap a frequency to the FM 100 kHz raster, but ONLY when it falls inside the
// FM broadcast band. Outside the FM band (ham, AM, airband, ...) snapping to a
// 0.1 MHz grid is wrong, so we leave the frequency exactly as picked. The UI
// checkbox can force-disable snapping entirely; its step field overrides the
// raster when set.
function snapFreq(hz: number): number {
  if (!ui.snapEnabled.checked) return Math.round(hz);

  const inFmBand = hz >= FM_BAND_LO && hz <= FM_BAND_HI;
  if (!inFmBand) return Math.round(hz);

  // The UI step field overrides the default 100 kHz raster when set to a
  // positive value; an explicit 0 means "no snap"; a blank/invalid field
  // falls back to the FM raster.
  const raw = ui.snapStep.value.trim();
  if (raw === "") return Math.round(hz / FM_RASTER_HZ) * FM_RASTER_HZ;
  const override = Number(raw) * 1_000_000;
  if (!Number.isFinite(override) || override <= 0) {
    return Math.round(hz); // step set to 0 explicitly means "no snap"
  }
  return Math.round(hz / override) * override;
}

interface TuneResponse {
  ok: boolean;
  config: { center_freq: number };
  tuned_freq: number;
  tune_offset: number;
  recentered: boolean;
}

async function tune(): Promise<void> {
  const rawInput = ui.frequencyInput.value;
  const oldTunedFreq = tunedFreq;
  // Snap BEFORE mutating the input so error recovery can restore the raw value.
  const snapped = snapFreq(Number(rawInput));
  const zoom = renderer?.zoom ?? 1;
  const payload = {
    freq: snapped,
    mode: selectedMode(ui.modeSegments),
    gain: gainValue(ui.gainSlider),
    fft_size: Number(ui.fftSelect.value),
    demod_bw: Number(ui.bwSlider.value) * 1000,
    squelch: Number(ui.squelchSlider.value),
    ppm: Number(ui.ppmSlider.value),
    rtl_agc: ui.rtlAgcCheck.checked,
    bias_tee: ui.biasTeeCheck.checked,
    direct_sampling: Number(ui.directSamplingSelect.value),
  };

  // Optimistic input update: show the snapped value immediately.
  ui.frequencyInput.value = String(snapped);
  tunedFreq = snapped; // instant marker update

  const response = await fetch('/api/tune', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    // Restore the pre-snap input so the UI never shows a value the
    // server rejected.
    ui.frequencyInput.value = rawInput;
    tunedFreq = oldTunedFreq;
    const message = await response.text();
    throw new Error(message);
  }

  // The backend decides whether this was an in-band retune (just move the
  // listening point, waterfall stays put) or an out-of-band hardware retune
  // (band recentres on the new frequency). Either way it tells us the actual
  // tuned frequency, which is what we mark and zoom around.
  const result = (await response.json()) as TuneResponse;
  tunedFreq = result.tuned_freq;

  // Optimistic update for immediate post-tune clicks: after setting a freq
  // that recentered the hardware + view, use the new tuned point as lastCenter
  // so the very next click computes the correct relative freq from the (new)
  // slice without waiting for the next waterfall frame.
  // IMPORTANT: only do this when the hardware recentered. For an in-band click
  // the slice centre hasn't moved — overwriting lastCenterFreq with the offset
  // tuned frequency would misalign the band plan / freq ruler until the next
  // waterfall frame arrives (~40 ms later).
  if (result.recentered) {
    lastCenterFreq = tunedFreq;
  }

  // Do NOT send a pan here. The zoom window stays where the user left it; the
  // backend only moves it (edge-follow) if the tuned point leaves the visible
  // window. Sending pan=0.5 here is exactly what made the view recentre on
  // every click. If the hardware recentred (out-of-band jump) reset zoom view.
  if (result.recentered) {
    renderer && (renderer.pan = 0.5);
    waterfallClient?.setView(zoom, 0.5);
  }

  ui.freqReadout.textContent = formatFrequency(tunedFreq);
  ui.modeReadout.textContent = payload.mode.toUpperCase();
  ui.bwReadout.textContent = `${payload.demod_bw >= 1000 ? (payload.demod_bw / 1000).toFixed(payload.demod_bw % 1000 ? 1 : 0) : payload.demod_bw} ${payload.demod_bw >= 1000 ? 'kHz' : 'Hz'}`;
  ui.fftReadout.textContent = `${payload.fft_size} FFT`;
  updateBwSliderPresets(payload.mode);
  updateBwReadout();
}

// Bandwidth presets per mode (kHz)
const BW_PRESETS: Record<string, { min: number; max: number; step: number; default: number }> = {
  wbfm: { min: 5, max: 250, step: 1, default: 15 },
  am:   { min: 1, max: 20, step: 0.5, default: 5 },
  usb:  { min: 0.2, max: 8, step: 0.1, default: 3 },
  lsb:  { min: 0.2, max: 8, step: 0.1, default: 3 },
  cw:   { min: 0.05, max: 2, step: 0.05, default: 0.5 },
};

function updateBwSliderPresets(mode: string): void {
  const m = (mode || 'wbfm').toLowerCase();
  const p = BW_PRESETS[m] || BW_PRESETS.am;
  ui.bwSlider.min = String(p.min);
  ui.bwSlider.max = String(p.max);
  ui.bwSlider.step = String(p.step);
  const cur = Number(ui.bwSlider.value);
  if (cur < p.min || cur > p.max) {
    ui.bwSlider.value = String(p.default);
  }
  ui.bwSlider.title = `Demodulation bandwidth in kHz. ${m.toUpperCase()} range: ${p.min}-${p.max} kHz`;
}

function updateBwReadout(): void {
  const bwKhz = Number(ui.bwSlider.value);
  const bwHz = Math.round(bwKhz * 1000);
  ui.bwOutput.textContent = `${bwKhz.toFixed(bwKhz < 1 ? 2 : bwKhz < 10 ? 1 : 0)} kHz`;
  ui.bwReadout.textContent = bwHz >= 1000 ? `${(bwHz / 1000).toFixed(bwHz % 1000 ? 1 : 0)} kHz` : `${bwHz} Hz`;
}

// Signal strength display helpers
function updateSignalMeter(signalDb: number, squelchOpen: boolean): void {
  const clamped = Math.max(-120, Math.min(-20, signalDb));
  const pct = ((clamped + 120) / 100) * 100;
  ui.sigBar.style.width = `${Math.max(0, Math.min(100, pct))}%`;
  ui.sigReadout.textContent = `${signalDb.toFixed(1)} dBFS`;
  ui.sigLevelDisplay.textContent = `${signalDb.toFixed(1)} dBFS`;
  const hue = signalDb > -40 ? 120 : signalDb > -70 ? 60 : 0;
  ui.sigBar.style.background = `hsl(${hue}, 80%, 50%)`;
  if (Number(ui.squelchSlider.value) <= -150) {
    ui.squelchLed.textContent = '●';
    ui.squelchLed.style.color = '#5a7070';
    ui.squelchLed.title = 'Squelch off';
  } else if (squelchOpen) {
    ui.squelchLed.textContent = '●';
    ui.squelchLed.style.color = '#26d6b0';
    ui.squelchLed.title = 'Squelch open (signal)';
  } else {
    ui.squelchLed.textContent = '●';
    ui.squelchLed.style.color = '#ff5a3d';
    ui.squelchLed.title = 'Squelch closed (muted)';
  }
}

async function refreshStatus(): Promise<void> {
  const response = await fetch('/api/status');
  if (!response.ok) return;
  const status = (await response.json()) as ApiStatus;
  ui.sourceLabel.textContent = status.running ? status.source : 'stopped';
  ui.clientReadout.textContent = `${status.waterfall_clients + status.audio_clients} clients`;
  ui.dropReadout.textContent = `${status.dropped_blocks} drops`;
  // Keep the local tuned frequency in sync with the server (it's authoritative
  // for the listening point, which may differ from the band centre).
  if (typeof status.tuned_freq === 'number') tunedFreq = status.tuned_freq;
  // Don't clobber the input while the user is typing in it — that caused the
  // "frequency snaps back to the old value" behaviour. Show the tuned
  // frequency, not the band centre.
  if (document.activeElement !== ui.frequencyInput) {
    ui.frequencyInput.value = String(Math.round(tunedFreq));
  }
  ui.freqReadout.textContent = formatFrequency(tunedFreq);
  ui.modeReadout.textContent = status.config.mode.toUpperCase();
  // Bandwidth readout
  const bw = status.config.demod_bw || 0;
  ui.bwReadout.textContent = bw >= 1000 ? `${(bw / 1000).toFixed(bw % 1000 ? 1 : 0)} kHz` : `${bw} Hz`;
  // Sample rate readout
  const sr = status.config.sample_rate || 0;
  ui.srReadout.textContent = sr >= 1_000_000 ? `${(sr / 1_000_000).toFixed(1)} MHz` : sr >= 1_000 ? `${(sr / 1_000).toFixed(0)} kHz` : `${sr} Hz`;
  ui.fftReadout.textContent = `${status.config.fft_size} FFT`;
  ui.fftSelect.value = String(status.config.fft_size);
  setMode(ui.modeSegments, status.config.mode);
  ui.gainSlider.value = status.config.gain === 'auto' ? '-1' : status.config.gain;
  ui.gainOutput.textContent = status.config.gain === '-1' ? 'auto' : status.config.gain;
  // Bandwidth slider
  updateBwSliderPresets(status.config.mode);
  if (bw > 0) {
    ui.bwSlider.value = String(bw / 1000);
  }
  updateBwReadout();
  // Squelch slider
  const squelchVal = typeof status.config.squelch === 'number' ? status.config.squelch : -160;
  ui.squelchSlider.value = String(squelchVal);
  ui.squelchOutput.textContent = squelchVal <= -150 ? 'off' : `${squelchVal.toFixed(0)} dBFS`;
  // Signal strength
  const sigDb = typeof status.signal_strength_db === 'number' ? status.signal_strength_db : -160;
  const sqOpen = status.squelch_open !== false;
  updateSignalMeter(sigDb, sqOpen);
  // PPM
  const ppm = typeof status.config.ppm === 'number' ? status.config.ppm : 0;
  ui.ppmSlider.value = String(ppm);
  ui.ppmOutput.textContent = ppm === 0 ? '0' : `${ppm > 0 ? '+' : ''}${ppm}`;
  // RTL AGC / Bias-T / Direct sampling
  ui.rtlAgcCheck.checked = !!status.config.rtl_agc;
  ui.biasTeeCheck.checked = !!status.config.bias_tee;
  ui.directSamplingSelect.value = String(status.config.direct_sampling || 0);
}

function isDraggingSlider(slider: HTMLInputElement): boolean {
  return slider.matches(':active');
}

function updateRangeOutputs(): void {
  ui.gainOutput.textContent = gainValue(ui.gainSlider);
  ui.floorOutput.textContent = ui.floorSlider.value;
  ui.ceilingOutput.textContent = ui.ceilingSlider.value;
}

function updateVolumeOutput(): void {
  const vol = Number(ui.volumeSlider.value);
  ui.volumeOutput.textContent = `${Math.round(vol * 100)}%`;
  if (gainNode) gainNode.gain.value = vol;
}

waterfallClient = connectWaterfall(
  (frame) => {
    lastSeq = frame.seq;
    lastCenterFreq = frame.centerFreq;
    lastSampleRate = frame.sampleRate;
    ensureRenderer(frame.fftSize).pushRow(frame.bins);
    lastBins = frame.bins;

    // Auto contrast: overwrite the manual sliders with signal-aware values.
    // This makes the waterfall "pop" automatically as you tune/zoom without
    // having to fiddle floor/ceiling all the time.
    // IMPORTANT: only update sliders when the user is NOT actively dragging
    // either slider — fighting a mid-drag position is jarring.
    if (ui.autoContrast.checked && !isDraggingSlider(ui.floorSlider) && !isDraggingSlider(ui.ceilingSlider)) {
      const { floor, ceiling } = computeAutoContrast(frame.bins);
      ui.floorSlider.value = String(floor);
      ui.ceilingSlider.value = String(ceiling);
      ui.floorOutput.textContent = String(floor);
      ui.ceilingOutput.textContent = String(ceiling);
    }

    // Keep the live drag shift in sync as re-centred slices arrive.
    // IMPORTANT: while the user is actively grabbing (dragging), the mouse
    // gesture owns the visual offset (set directly in the mousemove handler for
    // instant response). Calling refresh here would fight the current mouse
    // position and cause the "weird distortion / jumping" while dragging.
    // Only converge the view when not actively dragging (or in post-release settling).
    if (!dragging) {
      refreshDragOffset();
    }
    // Do NOT overwrite the tuned freq readout with the slice center every frame.
    // The main readout should reflect the demod tuned point (updated via tune response + status poll).
    // The visible range is shown in the freq strip / bandplan using canvasXToFreq.
    ui.fftReadout.textContent = `${frame.fftSize} FFT`;
  },
  (state) => {
    ui.sourceLabel.textContent = state;
  },
);

// Frequency at the CENTRE of what is currently on screen. Normally this is the
// slice centre from the latest frame, but WHILE dragging the shader slides the
// image by renderer.dragOffset (screen-fraction) before the backend re-centres.
// We mirror that same shift here so the frequency ruler + band plan track the
// dragged image instead of staying pinned to the not-yet-updated slice centre
// (which made the strip look offset during a drag).
function viewCenterFreq(): number {
  const dragShift = (renderer?.currentDragOffset ?? 0) * lastSampleRate;
  return lastCenterFreq - dragShift;
}

// Map a normalised canvas X (0..1, left..right) to an absolute frequency.
// With server-side zoom the latest frame's centerFreq/sampleRate already
// describe exactly the on-screen slice, so the canvas maps linearly across it.
function canvasXToFreq(x01: number): number {
  return viewCenterFreq() + (x01 - 0.5) * lastSampleRate;
}

// Inverse: absolute frequency -> normalised canvas X (may fall outside 0..1).
function freqToCanvasX(hz: number): number {
  return 0.5 + (hz - viewCenterFreq()) / lastSampleRate;
}

// Demod passband edges (Hz) relative to the tuned carrier, per mode. Uses
// the actual bandwidth slider value when available, with sensible defaults.
function modeBand(mode: string): { lo: number; hi: number } {
  const bwHz = Math.round((Number(ui.bwSlider.value) || 0) * 1000);
  switch (mode) {
    case 'wbfm': return { lo: -bwHz, hi: bwHz };
    case 'am': return { lo: -bwHz, hi: bwHz };
    case 'usb': return { lo: 0, hi: bwHz || 3_000 };
    case 'lsb': return { lo: -(bwHz || 3_000), hi: 0 };
    case 'cw': return { lo: -Math.max(50, Math.min(bwHz, 2000)) / 2, hi: Math.max(50, Math.min(bwHz, 2000)) / 2 };
    default: return { lo: -5_000, hi: 5_000 };
  }
}

// Band-plan strip: coloured service segments (40m, 31m, FM, AM, airband, ...)
// mapped onto the currently visible frequency window. Uses canvasXToFreq so it
// scrolls and zooms in lockstep with the waterfall, KiwiSDR-style.
function drawBandPlan(): void {
  const cv = ui.bandPlan;
  const ctx = cv.getContext('2d');
  if (!ctx) return;
  const ratio = window.devicePixelRatio || 1;
  const bw = Math.max(1, Math.floor(cv.clientWidth * ratio));
  const bh = Math.max(1, Math.floor(cv.clientHeight * ratio));
  if (cv.width !== bw || cv.height !== bh) {
    cv.width = bw;
    cv.height = bh;
  }
  const { width: w, height: h } = cv;
  ctx.clearRect(0, 0, w, h);

  const fLo = canvasXToFreq(0);
  const fHi = canvasXToFreq(1);
  const bands = bandsInRange(Math.min(fLo, fHi), Math.max(fLo, fHi));

  ctx.font = '10px monospace';
  ctx.textBaseline = 'middle';

  if (bands.length === 0) {
    ctx.fillStyle = '#566';
    ctx.textAlign = 'center';
    ctx.fillText('— no band-plan entries in view —', w / 2, h / 2);
    return;
  }

  for (const band of bands) {
    const x0 = Math.max(0, freqToCanvasX(band.lo) * w);
    const x1 = Math.min(w, freqToCanvasX(band.hi) * w);
    const bw = Math.max(1, x1 - x0);
    const c = BAND_COLORS[band.kind];

    ctx.fillStyle = c.fill;
    ctx.fillRect(x0, 0, bw, h);
    // Edges.
    ctx.fillStyle = c.text;
    ctx.globalAlpha = 0.5;
    ctx.fillRect(x0, 0, 1, h);
    ctx.fillRect(x1 - 1, 0, 1, h);
    ctx.globalAlpha = 1;

    // Label, only if it fits in the segment.
    const tw = ctx.measureText(band.label).width;
    if (bw > tw + 6) {
      ctx.fillStyle = c.text;
      ctx.textAlign = 'center';
      ctx.fillText(band.label, (x0 + x1) / 2, h / 2 + 0.5);
    }
  }
}

function drawFreqStrip(): void {
  const strip = ui.freqStrip;
  const ctx = strip.getContext('2d');
  if (!ctx) return;
  const ratio = window.devicePixelRatio || 1;
  const sw = Math.max(1, Math.floor(strip.clientWidth * ratio));
  const sh = Math.max(1, Math.floor(strip.clientHeight * ratio));
  if (strip.width !== sw || strip.height !== sh) {
    strip.width = sw;
    strip.height = sh;
  }
  ctx.clearRect(0, 0, strip.width, strip.height);

  const freqStart = canvasXToFreq(0);
  const freqEnd = canvasXToFreq(1);

  const steps = 10;
  ctx.fillStyle = '#9facaa';
  ctx.font = '10px monospace';
  for (let i = 0; i <= steps; i++) {
    const x = (i / steps) * strip.width;
    const freq = freqStart + (i / steps) * (freqEnd - freqStart);
    ctx.fillRect(x, strip.height - 8, 1, 8);
    ctx.textAlign = i === 0 ? 'left' : i === steps ? 'right' : 'center';
    ctx.fillText(formatFrequency(freq), x, strip.height - 12);
  }

  drawBandMarker(ctx, strip.width, strip.height);
}

// Draw the demod passband + mode label at the tuned carrier.
function drawBandMarker(ctx: CanvasRenderingContext2D, width: number, height: number): void {
  // Mark the frequency we're actually listening to (which may sit off-centre
  // on the waterfall now that click-to-tune doesn't recentre the band).
  const carrier = tunedFreq;
  const mode = selectedMode(ui.modeSegments);
  const band = modeBand(mode);

  const xLo = freqToCanvasX(carrier + band.lo) * width;
  const xHi = freqToCanvasX(carrier + band.hi) * width;
  const xCar = freqToCanvasX(carrier) * width;

  // Passband shading.
  ctx.fillStyle = 'rgba(38, 214, 176, 0.18)';
  ctx.fillRect(xLo, 0, Math.max(1, xHi - xLo), height - 8);

  // Passband edges.
  ctx.fillStyle = 'rgba(38, 214, 176, 0.55)';
  ctx.fillRect(xLo, 0, 1, height - 8);
  ctx.fillRect(xHi, 0, 1, height - 8);

  // Carrier line.
  ctx.fillStyle = '#26d6b0';
  ctx.fillRect(xCar - 0.5, 0, 1.5, height - 8);

  // Mode + bandwidth label above the carrier.
  const bw = band.hi - band.lo;
  const bwText = bw >= 1000 ? `${(bw / 1000).toFixed(bw % 1000 ? 1 : 0)} kHz` : `${bw} Hz`;
  const label = `${mode.toUpperCase()} · ${bwText}`;
  ctx.font = 'bold 10px monospace';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  const labelX = Math.min(width - 4, Math.max(4, xCar));
  // Backing box for legibility.
  const tw = ctx.measureText(label).width;
  ctx.fillStyle = 'rgba(11, 14, 17, 0.85)';
  ctx.fillRect(labelX - tw / 2 - 4, 1, tw + 8, 13);
  ctx.fillStyle = '#26d6b0';
  ctx.fillText(label, labelX, 3);
  ctx.textBaseline = 'alphabetic';
}

function draw(): void {
  const floor = Number(ui.floorSlider.value);
  const ceiling = Math.max(floor + 1, Number(ui.ceilingSlider.value));
  const r = ensureRenderer(currentFft);
  r.draw(floor, ceiling);

  drawBandPlan();
  drawFreqStrip();

  // Update zoom readout
  if (r.zoom > 1) {
    ui.zoomReadout.textContent = `${r.zoom.toFixed(1)}x zoom`;
    ui.zoomReadout.style.display = '';
  } else {
    ui.zoomReadout.style.display = 'none';
  }

  const now = performance.now();
  if (now - lastFpsAt >= 1000) {
    ui.frameReadout.textContent = `${frameCount} fps / seq ${lastSeq}`;
    frameCount = 0;
    lastFpsAt = now;
  }
  frameCount += 1;

  requestAnimationFrame(draw);
}

ui.tuneButton.addEventListener('click', () => {
  void tune().catch((error: unknown) => {
    ui.sourceLabel.textContent = error instanceof Error ? error.message : 'tune failed';
  });
});

ui.frequencyInput.addEventListener('keydown', (event) => {
  if (event.key === 'Enter') void tune();
});

// Click-to-tune: a click on the waterfall moves the listening point to the
// frequency under the cursor WITHOUT recentring the band -- the waterfall stays
// put and the tuning cursor moves to where you clicked. We track the press
// position only so a sloppy click that drifts a few pixels still tunes, while a
// real drag (>5px) is ignored.
let pressX = 0;
let pressY = 0;
let mousedownOnCanvas = false; // so we can reliably detect a click that started on the waterfall even if the mouseup target is not exactly the canvas element
// Drag-to-pan state. Active only when zoomed in (>1x) -- at 1x the whole band is
// already on screen so there's nothing to pan to. While dragging we slide the
// current texture locally (renderer.setDragOffset) for an instant, lag-free feel
// and stream the new view centre to the backend as an absolute frequency; on
// release the backend delivers a re-centred slice and we clear the local shift.
let dragging = false;
let dragSettling = false;  // true after release until the slice reaches the
                           // committed centre (offset converges to ~0)
let dragStartX = 0;
let dragStartCenter = 0;   // slice centre freq (Hz) at the moment the grab began
let dragStartRate = 0;     // slice width (Hz) at the moment the grab began
let dragDesiredCenter = 0; // where the mouse currently wants the view centred
// maximum safe local drag offset before committing the pan to the backend
const DRAG_COMMIT_THRESHOLD = 0.3;

// After mouseup, smoothly converges the local texture shift to zero as the
// backend delivers re-centred slices. Each incoming frame nudges the offset
// closer to zero because lastCenterFreq approaches dragDesiredCenter; once
// the difference is negligible we pin the shift to exactly 0 and stop.
function refreshDragOffset(): void {
  if (!renderer || lastSampleRate <= 0) return;
  if (!dragging && !dragSettling) return;
  const offset = (lastCenterFreq - dragDesiredCenter) / lastSampleRate;
  renderer.setDragOffset(offset);
  if (!dragging && Math.abs(offset) < 0.002) {
    renderer.setDragOffset(0);
    dragSettling = false;
  }
}

// A press ARMS a potential drag; the drag only ACTIVATES once the pointer moves
// past a small threshold (so a click still tunes). This arm-then-activate split
// avoids depending on a zoom read at the instant of mousedown.
let dragArmed = false;
ui.canvas.addEventListener('mousedown', (event) => {
  if (event.button !== 0) return;
  mousedownOnCanvas = true;
  pressX = event.clientX;
  pressY = event.clientY;
  dragArmed = true;
  dragStartX = event.clientX;
});

window.addEventListener('mousemove', (event) => {
  // Activate the drag the first time the pointer moves >4px while armed and
  // zoomed in. (At 1x there's nothing to pan to, so we leave it to tune-click.)
  if (dragArmed && !dragging) {
    const moved = Math.hypot(event.clientX - pressX, event.clientY - pressY);
    if (moved > 4 && canPanView()) {
      dragging = true;
      dragSettling = false;
      dragStartX = pressX;
      dragStartCenter = lastCenterFreq;
      dragStartRate = lastSampleRate;
      dragDesiredCenter = lastCenterFreq;
      ui.canvas.style.cursor = 'grabbing';
    }
  }
  if (!dragging || !renderer) return;

  const rect = ui.canvas.getBoundingClientRect();

  // If the hardware capture rate changed mid-drag, re-anchor everything so the
  // gesture stays 1:1 and can't compute a wildly off-band centre.
  if (lastSampleRate > 0 && lastSampleRate !== dragStartRate) {
    dragStartX = event.clientX;
    dragStartCenter = lastCenterFreq;
    dragStartRate = lastSampleRate;
  }

  const f = (event.clientX - dragStartX) / Math.max(1, rect.width); // screen-fraction
  // Drag RIGHT (f > 0) reveals LOWER frequencies, so the view centre decreases.
  dragDesiredCenter = dragStartCenter - f * dragStartRate;
  // Slide the on-screen texture with the mouse, INSTANTLY. The offset is purely
  // local (measured from the last commit or drag start), so the texture never
  // shifts more than DRAG_COMMIT_THRESHOLD away from the on-screen data, and
  // the black-band-at-the-edge artifact is bounded to ≤ threshold screen-fraction.
  renderer.setDragOffset(f);

  // Mid-drag pan commit: when the local texture shift approaches the edge of the
  // captured slice, commit the current view centre to the backend and reset the
  // local offset. The visual centre stays constant because we update
  // lastCenterFreq + re-anchor the drag baseline at the same time:
  //   before: center = lastCenterFreq - f * rate
  //   after:  center = dragDesiredCenter - 0 * rate = dragDesiredCenter
  // Since dragDesiredCenter = lastCenterFreq - f * rate, they are the same.
  // The very next frame from the server will already be centred at the committed
  // centre, so no jump-back on the next local shift.
  if (dragging && waterfallClient && Math.abs(f) > DRAG_COMMIT_THRESHOLD) {
    waterfallClient.panToHz(dragDesiredCenter);
    lastCenterFreq = dragDesiredCenter;
    dragStartCenter = dragDesiredCenter;
    dragStartX = event.clientX;
    renderer.setDragOffset(0);
  }
});

window.addEventListener('mouseup', (event) => {
  dragArmed = false;
  if (dragging) {
    dragging = false;
    dragSettling = true; // keep converging the local shift after release
    ui.canvas.style.cursor = '';
    // Commit the final centre. The local shift is NOT force-cleared here; it
    // converges to zero as re-centred frames arrive (see frame handler), so
    // there's no jump-back flash.
    waterfallClient?.panToHz(dragDesiredCenter);
    // Do NOT optimistically update lastCenterFreq here — that would make the
    // very next frame compute offset=0 in refreshDragOffset and terminate
    // settling immediately, snapping the texture back before the backend
    // responds. The settling loop handles convergence naturally as re-centred
    // frames arrive, and viewCenterFreq() still returns dragDesiredCenter
    // during settling (since dragOffset * rate = lastCenterFreq - dragDesiredCenter),
    // so click-to-tune after drag works correctly.
    mousedownOnCanvas = false;
    return; // a drag is never a tune click
  }

  const moved = Math.hypot(event.clientX - pressX, event.clientY - pressY);
  if (moved > 5) return; // it was a drag/pan, not a tune click

  // Click-to-tune must have started on the canvas (we use this flag because
  // after a drag or when the cursor is on the edge/border, event.target on
  // window mouseup may not be exactly the canvas).
  const startedOnCanvas = mousedownOnCanvas;
  mousedownOnCanvas = false;
  if (!startedOnCanvas) return;

  const rect = ui.canvas.getBoundingClientRect();
  const x01 = Math.min(1, Math.max(0, (event.clientX - rect.left) / rect.width));
  const freq = snapFreq(canvasXToFreq(x01));

  // Save for rollback before the optimistic update
  const oldInput = ui.frequencyInput.value;
  const oldTuned = tunedFreq;

  // The latest frame's centerFreq/sampleRate describe exactly the on-screen
  // slice, so canvasXToFreq maps the click linearly across it. tune() sends
  // this to the backend, which moves the listening point there (and only
  // recentres the hardware if it falls outside the captured band).
  ui.frequencyInput.value = String(freq);
  tunedFreq = freq; // Instant marker placement so the band cursor jumps to where you clicked without waiting for the round-trip.
  void tune().catch((error: unknown) => {
    ui.frequencyInput.value = oldInput;
    tunedFreq = oldTuned;
    ui.sourceLabel.textContent = error instanceof Error ? error.message : 'tune failed';
  });
});

// Show a grab cursor when hovering the waterfall while zoomed (affordance).
ui.canvas.addEventListener('mousemove', () => {
  if (dragging) return;
  ui.canvas.style.cursor = canPanView() ? 'grab' : '';
});

ui.modeSegments.addEventListener('click', (event) => {
  const target = event.target as HTMLElement;
  const button = target.closest<HTMLButtonElement>('button[data-mode]');
  if (!button) return;
  const newMode = button.dataset.mode ?? 'wbfm';
  const oldMode = selectedMode(ui.modeSegments);
  if (newMode === oldMode) return;
  setMode(ui.modeSegments, newMode);
  tune().catch((error: unknown) => {
    // Revert the button highlight so the UI doesn't lie about what mode
    // the server is actually using (the status poll will catch up, but
    // reverting immediately is better UX than a ~1.5 s desync window).
    setMode(ui.modeSegments, oldMode);
    ui.sourceLabel.textContent = error instanceof Error ? error.message : 'tune failed';
  });
});

ui.gainSlider.addEventListener('input', updateRangeOutputs);
ui.gainSlider.addEventListener('change', () => void tune());

// Bandwidth slider
ui.bwSlider.addEventListener('input', updateBwReadout);
ui.bwSlider.addEventListener('change', () => void tune());

// Squelch slider
ui.squelchSlider.addEventListener('input', () => {
  const v = Number(ui.squelchSlider.value);
  ui.squelchOutput.textContent = v <= -150 ? 'off' : `${v.toFixed(0)} dBFS`;
});
ui.squelchSlider.addEventListener('change', () => void tune());

// PPM slider
ui.ppmSlider.addEventListener('input', () => {
  const v = Number(ui.ppmSlider.value);
  ui.ppmOutput.textContent = v === 0 ? '0' : `${v > 0 ? '+' : ''}${v}`;
});
ui.ppmSlider.addEventListener('change', () => void tune());

// RTL AGC / Bias-T / Direct sampling
ui.rtlAgcCheck.addEventListener('change', () => void tune());
ui.biasTeeCheck.addEventListener('change', () => void tune());
ui.directSamplingSelect.addEventListener('change', () => void tune());

// Dragging the zoom slider drives the SAME zoom as the mouse wheel. setZoom
// applies it, sends the view to the backend and (via onZoomChange) updates the
// readout. We update the readout here too for instant feedback while dragging.
ui.zoomSlider.addEventListener('input', () => {
  const zoom = sliderToZoom(Number(ui.zoomSlider.value));
  ui.zoomOutput.textContent = `${zoom.toFixed(1)}x`;
  ensureRenderer(currentFft).setZoom(zoom);
});

ui.volumeSlider.addEventListener('input', updateVolumeOutput);

function detectDevice(): void {
  if (/Mobi|Android/i.test(navigator.userAgent)) ui.fftSelect.value = '8192';
}
detectDevice();
ui.fftSelect.addEventListener('change', () => void tune());
ui.paletteSelect.addEventListener('change', () => renderer?.setColorMap(selectedPalette(ui.paletteSelect)));
ui.floorSlider.addEventListener('input', updateRangeOutputs);
ui.ceilingSlider.addEventListener('input', updateRangeOutputs);

ui.autoContrast.addEventListener('change', () => {
  if (ui.autoContrast.checked && lastBins) {
    const { floor, ceiling } = computeAutoContrast(lastBins);
    ui.floorSlider.value = String(floor);
    ui.ceilingSlider.value = String(ceiling);
    ui.floorOutput.textContent = String(floor);
    ui.ceilingOutput.textContent = String(ceiling);
  }
});

ui.audioButton.addEventListener('click', async () => {
  if (audio) {
    audio.close();
    audio = undefined;
    gainNode = undefined;
    ui.audioButton.innerHTML = '<span data-lucide="volume-2"></span>Start Audio';
    refreshIcons();
    return;
  }

  if (audioStarting) return;
  audioStarting = true;
  ui.audioButton.disabled = true;
  try {
    const client = await connectAudio();
    await client.ctx.resume();
    audio = client;

    // Insert a gain node between worklet and destination for volume control
    gainNode = audio.ctx.createGain();
    gainNode.gain.value = Number(ui.volumeSlider.value);
    audio.node.disconnect();
    audio.node.connect(gainNode);
    gainNode.connect(audio.ctx.destination);

    ui.audioButton.innerHTML = '<span data-lucide="volume-x"></span>Stop Audio';
    refreshIcons();
  } catch (error) {
    ui.sourceLabel.textContent = error instanceof Error ? error.message : 'audio failed';
  } finally {
    audioStarting = false;
    ui.audioButton.disabled = false;
  }
});

// --------------------------------------------------------------------------
// Keyboard shortcuts
// --------------------------------------------------------------------------
window.addEventListener('keydown', (event) => {
  // Don't intercept when the user is typing in the frequency input
  if (document.activeElement === ui.frequencyInput || document.activeElement === ui.snapStep) return;

  switch (event.key) {
    case ' ':
      // Space: toggle audio
      event.preventDefault();
      ui.audioButton.click();
      break;
    case '+':
    case '=':
      // Zoom in
      event.preventDefault();
      ensureRenderer(currentFft).setZoom((renderer?.zoom ?? 1) * 1.25);
      break;
    case '-':
    case '_':
      // Zoom out
      event.preventDefault();
      ensureRenderer(currentFft).setZoom((renderer?.zoom ?? 1) * 0.8);
      break;
    case 'ArrowUp':
    case 'ArrowRight': {
      // Nudge frequency up by the snap step (or 1 kHz if no snap)
      event.preventDefault();
      const raw = ui.snapStep.value.trim();
      let step = raw ? Number(raw) * 1_000_000 : 1000;
      if (!Number.isFinite(step) || step <= 0) step = 1000;
      const next = Math.round((Number(ui.frequencyInput.value) || 0) + step);
      ui.frequencyInput.value = String(next);
      void tune().catch((error: unknown) => {
        ui.sourceLabel.textContent = error instanceof Error ? error.message : 'tune failed';
      });
      break;
    }
    case 'ArrowDown':
    case 'ArrowLeft': {
      // Nudge frequency down by the snap step
      event.preventDefault();
      const raw = ui.snapStep.value.trim();
      let step = raw ? Number(raw) * 1_000_000 : 1000;
      if (!Number.isFinite(step) || step <= 0) step = 1000;
      const next = Math.max(1, Math.round((Number(ui.frequencyInput.value) || 0) - step));
      ui.frequencyInput.value = String(next);
      void tune().catch((error: unknown) => {
        ui.sourceLabel.textContent = error instanceof Error ? error.message : 'tune failed';
      });
      break;
    }
    case 'Enter':
      // Enter on the freq input is already handled by keydown on the input
      break;
  }
});

updateRangeOutputs();
updateVolumeOutput();
void refreshStatus();
window.setInterval(() => void refreshStatus(), 1500);
requestAnimationFrame(draw);
