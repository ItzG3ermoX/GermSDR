import { createIcons, icons } from 'lucide';
import { ColorMapName } from './colormap';

export interface UiElements {
  canvas: HTMLCanvasElement;
  freqStrip: HTMLCanvasElement;
  bandPlan: HTMLCanvasElement;
  frequencyInput: HTMLInputElement;
  tuneButton: HTMLButtonElement;
  modeSegments: HTMLElement;
  bwSlider: HTMLInputElement;
  bwOutput: HTMLOutputElement;
  gainSlider: HTMLInputElement;
  gainOutput: HTMLOutputElement;
  squelchSlider: HTMLInputElement;
  squelchOutput: HTMLOutputElement;
  squelchLed: HTMLElement;
  sigLevelDisplay: HTMLElement;
  ppmSlider: HTMLInputElement;
  ppmOutput: HTMLOutputElement;
  zoomSlider: HTMLInputElement;
  zoomOutput: HTMLOutputElement;
  volumeSlider: HTMLInputElement;
  volumeOutput: HTMLOutputElement;
  fftSelect: HTMLSelectElement;
  paletteSelect: HTMLSelectElement;
  floorSlider: HTMLInputElement;
  floorOutput: HTMLOutputElement;
  ceilingSlider: HTMLInputElement;
  ceilingOutput: HTMLOutputElement;
  autoContrast: HTMLInputElement;
  snapStep: HTMLInputElement;
  snapEnabled: HTMLInputElement;
  audioButton: HTMLButtonElement;
  sourceLabel: HTMLElement;
  freqReadout: HTMLElement;
  modeReadout: HTMLElement;
  bwReadout: HTMLElement;
  srReadout: HTMLElement;
  fftReadout: HTMLElement;
  sigReadout: HTMLElement;
  sigBar: HTMLElement;
  sigMeter: HTMLElement;
  frameReadout: HTMLElement;
  clientReadout: HTMLElement;
  dropReadout: HTMLElement;
  zoomReadout: HTMLElement;
  rtlAgcCheck: HTMLInputElement;
  biasTeeCheck: HTMLInputElement;
  directSamplingSelect: HTMLSelectElement;
}

function byId<T extends HTMLElement>(id: string): T {
  const element = document.getElementById(id);
  if (!element) throw new Error(`missing element #${id}`);
  return element as T;
}

export function getUi(): UiElements {
  refreshIcons();
  return {
    canvas: byId<HTMLCanvasElement>('waterfall'),
    freqStrip: byId<HTMLCanvasElement>('freqStrip'),
    bandPlan: byId<HTMLCanvasElement>('bandPlan'),
    frequencyInput: byId<HTMLInputElement>('frequencyInput'),
    tuneButton: byId<HTMLButtonElement>('tuneButton'),
    modeSegments: byId<HTMLElement>('modeSegments'),
    bwSlider: byId<HTMLInputElement>('bwSlider'),
    bwOutput: byId<HTMLOutputElement>('bwOutput'),
    gainSlider: byId<HTMLInputElement>('gainSlider'),
    gainOutput: byId<HTMLOutputElement>('gainOutput'),
    squelchSlider: byId<HTMLInputElement>('squelchSlider'),
    squelchOutput: byId<HTMLOutputElement>('squelchOutput'),
    squelchLed: byId<HTMLElement>('squelchLed'),
    sigLevelDisplay: byId<HTMLElement>('sigLevelDisplay'),
    ppmSlider: byId<HTMLInputElement>('ppmSlider'),
    ppmOutput: byId<HTMLOutputElement>('ppmOutput'),
    zoomSlider: byId<HTMLInputElement>('zoomSlider'),
    zoomOutput: byId<HTMLOutputElement>('zoomOutput'),
    volumeSlider: byId<HTMLInputElement>('volumeSlider'),
    volumeOutput: byId<HTMLOutputElement>('volumeOutput'),
    fftSelect: byId<HTMLSelectElement>('fftSelect'),
    paletteSelect: byId<HTMLSelectElement>('paletteSelect'),
    floorSlider: byId<HTMLInputElement>('floorSlider'),
    floorOutput: byId<HTMLOutputElement>('floorOutput'),
    ceilingSlider: byId<HTMLInputElement>('ceilingSlider'),
    ceilingOutput: byId<HTMLOutputElement>('ceilingOutput'),
    autoContrast: byId<HTMLInputElement>('autoContrast'),
    snapStep: byId<HTMLInputElement>('snapStep'),
    snapEnabled: byId<HTMLInputElement>('snapEnabled'),
    audioButton: byId<HTMLButtonElement>('audioButton'),
    sourceLabel: byId<HTMLElement>('sourceLabel'),
    freqReadout: byId<HTMLElement>('freqReadout'),
    modeReadout: byId<HTMLElement>('modeReadout'),
    bwReadout: byId<HTMLElement>('bwReadout'),
    srReadout: byId<HTMLElement>('srReadout'),
    fftReadout: byId<HTMLElement>('fftReadout'),
    sigReadout: byId<HTMLElement>('sigReadout'),
    sigBar: byId<HTMLElement>('sigBar'),
    sigMeter: byId<HTMLElement>('sigMeter'),
    frameReadout: byId<HTMLElement>('frameReadout'),
    clientReadout: byId<HTMLElement>('clientReadout'),
    dropReadout: byId<HTMLElement>('dropReadout'),
    zoomReadout: byId<HTMLElement>('zoomReadout'),
    rtlAgcCheck: byId<HTMLInputElement>('rtlAgcCheck'),
    biasTeeCheck: byId<HTMLInputElement>('biasTeeCheck'),
    directSamplingSelect: byId<HTMLSelectElement>('directSamplingSelect'),
  };
}

export function refreshIcons(): void {
  createIcons({ icons });
}

export function formatFrequency(hz: number): string {
  if (hz >= 1_000_000) return `${(hz / 1_000_000).toFixed(3)} MHz`;
  if (hz >= 1_000) return `${(hz / 1_000).toFixed(1)} kHz`;
  return `${hz.toFixed(0)} Hz`;
}

export function selectedMode(root: HTMLElement): string {
  return root.querySelector<HTMLButtonElement>('button.active')?.dataset.mode ?? 'wbfm';
}

export function setMode(root: HTMLElement, mode: string): void {
  root.querySelectorAll<HTMLButtonElement>('button').forEach((button) => {
    button.classList.toggle('active', button.dataset.mode === mode);
  });
}

export function selectedPalette(select: HTMLSelectElement): ColorMapName {
  const value = select.value as ColorMapName;
  if (['spectrum', 'inferno', 'viridis', 'mono'].includes(value)) return value;
  return 'spectrum';
}

export function gainValue(slider: HTMLInputElement): string {
  return Number(slider.value) < 0 ? 'auto' : slider.value;
}
