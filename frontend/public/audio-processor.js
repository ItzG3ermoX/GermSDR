class SdrAudioProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.chunks = [];
    this.offset = 0;
    this.maxChunks = 24;

    this.port.onmessage = (event) => {
      if (!event.data || !event.data.pcm) {
        return;
      }
      this.chunks.push(new Float32Array(event.data.pcm));
      while (this.chunks.length > this.maxChunks) {
        this.chunks.shift();
        this.offset = 0;
      }
    };
  }

  nextSample() {
    while (this.chunks.length) {
      const chunk = this.chunks[0];
      if (this.offset < chunk.length) {
        return chunk[this.offset++];
      }
      this.chunks.shift();
      this.offset = 0;
    }
    return 0;
  }

  process(_inputs, outputs) {
    const output = outputs[0];
    const left = output[0];
    const right = output[1] || output[0];

    for (let i = 0; i < left.length; i++) {
      const value = this.nextSample();
      left[i] = value;
      right[i] = value;
    }

    return true;
  }
}

registerProcessor('sdr-audio', SdrAudioProcessor);

