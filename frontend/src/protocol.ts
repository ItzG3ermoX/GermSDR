export const HEADER_BYTES = 20;

export interface WaterfallFrame {
  seq: number;
  centerFreq: number;
  sampleRate: number;
  fftSize: number;
  flags: number;
  bins: Float32Array;
}

export function parseWaterfallFrame(buffer: ArrayBuffer): WaterfallFrame {
  if (buffer.byteLength < HEADER_BYTES) {
    throw new Error(`waterfall frame too short: ${buffer.byteLength} bytes`);
  }

  const view = new DataView(buffer);
  const seq = view.getUint32(0, false);
  const centerFreq = view.getFloat64(4, false);
  const sampleRate = view.getFloat32(12, false);
  const fftSize = view.getUint16(16, false);
  const flags = view.getUint16(18, false);

  const expectedBytes = HEADER_BYTES + fftSize * Float32Array.BYTES_PER_ELEMENT;
  if (buffer.byteLength < expectedBytes) {
    throw new Error(`waterfall payload short: expected ${expectedBytes}, got ${buffer.byteLength}`);
  }

  return {
    seq,
    centerFreq,
    sampleRate,
    fftSize,
    flags,
    bins: new Float32Array(buffer, HEADER_BYTES, fftSize),
  };
}

