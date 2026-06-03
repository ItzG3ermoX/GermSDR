import { createIcons, icons } from 'lucide';
function byId(id) {
    const element = document.getElementById(id);
    if (!element) {
        throw new Error(`missing element #${id}`);
    }
    return element;
}
export function getUi() {
    refreshIcons();
    return {
        canvas: byId('waterfall'),
        frequencyInput: byId('frequencyInput'),
        tuneButton: byId('tuneButton'),
        modeSegments: byId('modeSegments'),
        gainSlider: byId('gainSlider'),
        gainOutput: byId('gainOutput'),
        fftSelect: byId('fftSelect'),
        paletteSelect: byId('paletteSelect'),
        floorSlider: byId('floorSlider'),
        floorOutput: byId('floorOutput'),
        ceilingSlider: byId('ceilingSlider'),
        ceilingOutput: byId('ceilingOutput'),
        audioButton: byId('audioButton'),
        sourceLabel: byId('sourceLabel'),
        freqReadout: byId('freqReadout'),
        modeReadout: byId('modeReadout'),
        fftReadout: byId('fftReadout'),
        frameReadout: byId('frameReadout'),
        clientReadout: byId('clientReadout'),
        dropReadout: byId('dropReadout'),
    };
}
export function refreshIcons() {
    createIcons({ icons });
}
export function formatFrequency(hz) {
    if (hz >= 1_000_000) {
        return `${(hz / 1_000_000).toFixed(3)} MHz`;
    }
    if (hz >= 1_000) {
        return `${(hz / 1_000).toFixed(1)} kHz`;
    }
    return `${hz.toFixed(0)} Hz`;
}
export function selectedMode(root) {
    return root.querySelector('button.active')?.dataset.mode ?? 'wbfm';
}
export function setMode(root, mode) {
    root.querySelectorAll('button').forEach((button) => {
        button.classList.toggle('active', button.dataset.mode === mode);
    });
}
export function selectedPalette(select) {
    const value = select.value;
    if (['spectrum', 'inferno', 'viridis', 'mono'].includes(value)) {
        return value;
    }
    return 'spectrum';
}
export function gainValue(slider) {
    return Number(slider.value) < 0 ? 'auto' : slider.value;
}
