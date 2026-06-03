const MAPS = {
    spectrum: [
        [0, 0, 0, 8],
        [0.16, 20, 58, 160],
        [0.34, 10, 194, 210],
        [0.52, 30, 185, 76],
        [0.72, 240, 210, 45],
        [0.88, 230, 64, 32],
        [1, 255, 255, 245],
    ],
    inferno: [
        [0, 3, 2, 18],
        [0.22, 66, 10, 104],
        [0.44, 147, 38, 103],
        [0.68, 229, 92, 45],
        [0.86, 249, 173, 55],
        [1, 252, 255, 164],
    ],
    viridis: [
        [0, 68, 1, 84],
        [0.25, 59, 82, 139],
        [0.5, 33, 145, 140],
        [0.75, 94, 201, 98],
        [1, 253, 231, 37],
    ],
    mono: [
        [0, 0, 0, 0],
        [0.38, 24, 44, 52],
        [0.72, 140, 178, 174],
        [1, 255, 255, 255],
    ],
};
function lerp(a, b, t) {
    return a + (b - a) * t;
}
export function makeColorMap(name) {
    const stops = MAPS[name] ?? MAPS.spectrum;
    const lut = new Uint8Array(256 * 4);
    for (let i = 0; i < 256; i++) {
        const t = i / 255;
        let lo = stops[0];
        let hi = stops[stops.length - 1];
        for (let j = 0; j < stops.length - 1; j++) {
            if (t >= stops[j][0] && t <= stops[j + 1][0]) {
                lo = stops[j];
                hi = stops[j + 1];
                break;
            }
        }
        const span = Math.max(0.0001, hi[0] - lo[0]);
        const k = (t - lo[0]) / span;
        lut[i * 4] = Math.round(lerp(lo[1], hi[1], k));
        lut[i * 4 + 1] = Math.round(lerp(lo[2], hi[2], k));
        lut[i * 4 + 2] = Math.round(lerp(lo[3], hi[3], k));
        lut[i * 4 + 3] = 255;
    }
    return lut;
}
