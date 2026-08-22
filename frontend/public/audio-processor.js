/**
 * GermSDR Audio Worklet - clean jitter buffer, no choppy playback
 */
class SdrAudioProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super(options);

    // Simple FIFO using a large pre-allocated ring buffer
    this._RING = 96000; // 2s at 48kHz
    this._buf = new Float32Array(this._RING);
    this._wr = 0;
    this._rd = 0;
    this._count = 0;

    // Don't start playing until we have ~80ms buffered (avoids immediate underrun)
    this._TARGET = 3840; // 80ms at 48kHz
    this._ready = false;
    this._last = 0; // last output sample, for click-free underrun fades

    this.port.onmessage = ({ data }) => {
      if (!data?.pcm) return;
      const src = new Float32Array(data.pcm);
      for (let i = 0; i < src.length; i++) {
        this._buf[this._wr] = src[i];
        this._wr = (this._wr + 1) % this._RING;
        if (this._count < this._RING) {
          this._count++;
        } else {
          // overflow: drop oldest
          this._rd = (this._rd + 1) % this._RING;
        }
      }
    };
  }

  process(_i, outputs) {
    const ch = outputs[0];
    const L = ch[0], R = ch[1] || ch[0];
    const n = L.length;

    // Wait for initial buffering
    if (!this._ready) {
      if (this._count >= this._TARGET) this._ready = true;
      else { L.fill(0); R.fill(0); return true; }
    }

    // Underrun — fade the last sample to zero instead of a hard cut (no click),
    // then rebuffer.
    if (this._count < n) {
      this._ready = false;
      for (let i = 0; i < n; i++) {
        const v = this._last * (1 - i / n);
        L[i] = v; R[i] = v;
      }
      this._last = 0;
      return true;
    }

    for (let i = 0; i < n; i++) {
      const v = this._buf[this._rd];
      this._rd = (this._rd + 1) % this._RING;
      this._count--;
      L[i] = v;
      R[i] = v;
    }
    this._last = L[n - 1];
    return true;
  }
}
registerProcessor('sdr-audio', SdrAudioProcessor);
