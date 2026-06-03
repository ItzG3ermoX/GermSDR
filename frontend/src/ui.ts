import { createIcons, icons } from 'lucide';
import { ColorMapName } from './colormap';

export interface UiElements {
  canvas: HTMLCanvasElement;
  frequencyInput: HTMLInputElement;
  tuneButton: HTMLButtonElement;
  modeSegments: HTMLElement;
  gainSlider: HTMLInputElement;
  gainOutput: HTMLOutputElement;
  fftSelect: HTMLSelectElement;
  paletteSelect: HTMLSelectElement;
  floorSlider: HTMLInputElement;
  floorOutput: HTMLOutputElement;
  ceilingSlider: HTMLInputElement;
  ceilingOutput: HTMLOutputElement;
  audioButton: HTMLButtonElement;
  sourceLabel: HTMLElement;
  freqReadout: HTMLElement;
  modeReadout: HTMLElement;
  fftReadout: HTMLElement;
  frameReadout: HTMLElement;
  clientReadout: HTMLElement;
  dropReadout: HTMLElement;
}

function byId<T extends HTMLElement>(id: string): T {
  const element = document.getElementById(id);
  if (!element) {
    throw new Error(`missing element #${id}`);
  }
  return element as T;
}

export function getUi(): UiElements {
  refreshIcons();
  return {
    canvas: byId<HTMLCanvasElement>('waterfall'),
    frequencyInput: byId<HTMLInputElement>('frequencyInput'),
    tuneButton: byId<HTMLButtonElement>('tuneButton'),
    modeSegments: byId<HTMLElement>('modeSegments'),
    gainSlider: byId<HTMLInputElement>('gainSlider'),
    gainOutput: byId<HTMLOutputElement>('gainOutput'),
    fftSelect: byId<HTMLSelectElement>('fftSelect'),
    paletteSelect: byId<HTMLSelectElement>('paletteSelect'),
    floorSlider: byId<HTMLInputElement>('floorSlider'),
    floorOutput: byId<HTMLOutputElement>('floorOutput'),
    ceilingSlider: byId<HTMLInputElement>('ceilingSlider'),
    ceilingOutput: byId<HTMLOutputElement>('ceilingOutput'),
    audioButton: byId<HTMLButtonElement>('audioButton'),
    sourceLabel: byId<HTMLElement>('sourceLabel'),
    freqReadout: byId<HTMLElement>('freqReadout'),
    modeReadout: byId<HTMLElement>('modeReadout'),
    fftReadout: byId<HTMLElement>('fftReadout'),
    frameReadout: byId<HTMLElement>('frameReadout'),
    clientReadout: byId<HTMLElement>('clientReadout'),
    dropReadout: byId<HTMLElement>('dropReadout'),
  };
}

export function refreshIcons(): void {
  createIcons({ icons });
}

export function formatFrequency(hz: number): string {
  if (hz >= 1_000_000) {
    return `${(hz / 1_000_000).toFixed(3)} MHz`;
  }
  if (hz >= 1_000) {
    return `${(hz / 1_000).toFixed(1)} kHz`;
  }
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
  if (['spectrum', 'inferno', 'viridis', 'mono'].includes(value)) {
    return value;
  }
  return 'spectrum';
}

export function gainValue(slider: HTMLInputElement): string {
  return Number(slider.value) < 0 ? 'auto' : slider.value;
}
