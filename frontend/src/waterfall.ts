import { ColorMapName, makeColorMap } from './colormap';

// Top-to-bottom waterfall: newest row at top.
// u_head is the write pointer (0..1). Rows above head are newer, below are older.
// We flip v_uv.y so y=0 is top of screen = newest data.
const VERT_SRC = `#version 300 es
  in vec2 a_pos;
  out vec2 v_uv;

  void main() {
    // Map clip coords [-1,1] -> uv [0,1], but flip Y so top of screen = uv.y=0
    v_uv = vec2(a_pos.x * 0.5 + 0.5, 0.5 - a_pos.y * 0.5);
    gl_Position = vec4(a_pos, 0.0, 1.0);
  }
`;

// Zoom-aware fragment shader: when u_zoom > 1 we sample a narrow band of freq
// and use NEAREST filtering override via explicit texel fetch for pixel-perfect sharpness.
const FRAG_SRC = `#version 300 es
  precision highp float;

  uniform sampler2D u_wf;
  uniform sampler2D u_lut;
  uniform float u_min;
  uniform float u_max;
  uniform float u_head;
  uniform float u_zoom;    // zoom factor >= 1
  uniform float u_pan;     // pan 0..1 (center of view)
  uniform float u_px;      // viewport width in device pixels
  uniform float u_drag;    // live drag offset in screen-fraction (-1..1); shifts
                           // the visible slice horizontally WHILE dragging so the
                           // image follows the mouse before the backend re-centres.

  in vec2 v_uv;
  out vec4 fragColor;

  // Read one texel's dB value (decode from the 0..255 R channel encoding).
  float sampleDb(ivec2 sz, int xi, int yi) {
    xi = clamp(xi, 0, sz.x - 1);
    yi = ((yi % sz.y) + sz.y) % sz.y;   // wrap time (ring buffer)
    return texelFetch(u_wf, ivec2(xi, yi), 0).r * 200.0 - 160.0;
  }

  void main() {
    // Map horizontal uv through zoom/pan window (server already delivers the
    // slice, so u_zoom is 1 / u_pan is 0.5 in practice -- kept for generality).
    float halfSpan = 0.5 / u_zoom;
    // Subtract u_drag so dragging the mouse RIGHT slides the image right (grab-
    // and-pull, like dragging a map).
    float freqX = (u_pan - u_drag) - halfSpan + v_uv.x / u_zoom;

    // Outside the captured slice (only possible while dragging, where freqX runs
    // off an edge): paint the colour-map floor instead of CLAMPING to the edge
    // texel. Clamping repeated the last column across the whole dragged-in
    // region, which read as a stretched smear on the side. A blank floor makes
    // it obvious that area has no data yet -- the backend fills it on release.
    if (freqX < 0.0 || freqX > 1.0) {
      fragColor = texture(u_lut, vec2(0.0, 0.5));
      return;
    }

    // Time axis: newest row at the TOP, scrolling downward as it ages.
    float timeY = mod(u_head - v_uv.y, 1.0);

    ivec2 sz = textureSize(u_wf, 0);
    int yi = int(floor(timeY * float(sz.y)));

    // PEAK (max-hold) sampling along frequency -- the key to a SHARP waterfall.
    // Several FFT bins can fall under one screen pixel; if we blended them
    // (bilinear) a 1-bin-wide carrier would be averaged into a soft grey blob.
    // Instead we take the MAX dB over exactly the bins this pixel covers, so a
    // narrow signal stays a crisp, full-brightness line -- like KiwiSDR. dFdx
    // gives us the exact change in frequency-bin coordinate per screen pixel,
    // which is robust regardless of zoom / DPR / viewport width.
    float fx = freqX * float(sz.x);                   // bin coordinate at pixel centre
    float binsPerPixel = max(abs(dFdx(fx)), abs(dFdy(fx)));
    int span = clamp(int(ceil(binsPerPixel)), 1, 32);
    float db = -200.0;
    int x0 = int(floor(fx - 0.5 * float(span)));
    for (int k = 0; k < 32; k++) {
      if (k >= span) break;
      db = max(db, sampleDb(sz, x0 + k, yi));
    }

    float t = clamp((db - u_min) / max(0.001, u_max - u_min), 0.0, 1.0);
    fragColor = texture(u_lut, vec2(t, 0.5));
  }
`;

const ENCODE_MIN_DB = -160;
const ENCODE_SPAN_DB = 200;

function compileShader(gl: WebGL2RenderingContext, type: number, source: string): WebGLShader {
  const shader = gl.createShader(type);
  if (!shader) throw new Error('failed to create WebGL shader');
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const message = gl.getShaderInfoLog(shader) ?? 'unknown shader compile error';
    gl.deleteShader(shader);
    throw new Error(message);
  }
  return shader;
}

function buildProgram(gl: WebGL2RenderingContext): WebGLProgram {
  const program = gl.createProgram();
  if (!program) throw new Error('failed to create WebGL program');
  const vert = compileShader(gl, gl.VERTEX_SHADER, VERT_SRC);
  const frag = compileShader(gl, gl.FRAGMENT_SHADER, FRAG_SRC);
  gl.attachShader(program, vert);
  gl.attachShader(program, frag);
  gl.linkProgram(program);
  gl.deleteShader(vert);
  gl.deleteShader(frag);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    const message = gl.getProgramInfoLog(program) ?? 'unknown WebGL link error';
    gl.deleteProgram(program);
    throw new Error(message);
  }
  return program;
}

export class WaterfallRenderer {
  private gl: WebGL2RenderingContext;
  private program: WebGLProgram;
  private wfTex: WebGLTexture;
  private lutTex: WebGLTexture;
  private vao: WebGLVertexArrayObject;
  private rowBuffer: Uint8Array;
  private writeRow = 0;
  private readonly sourceWidth: number;
  private readonly textureWidth: number;
  private readonly height: number;

  // Logical zoom / pan: what the USER wants to see. With server-side zoom these
  // are sent to the backend, which returns exactly that slice -- so the shader
  // itself draws the incoming texture at 1x (see draw()).
  zoom = 1.0;   // 1 = full view, 2 = 2x zoom, etc.
  pan = 0.5;    // 0..1, centre of the zoomed window

  // Live horizontal drag offset (screen-fraction, -1..1). Non-zero only WHILE the
  // user is grabbing the waterfall; the shader uses it to slide the current
  // texture so the drag feels instant. Committed into `pan` (and reset to 0) on
  // release, when the backend re-centres the captured slice.
  private dragOffset = 0;

  // Notified whenever the user changes the view. Pan is optional: omitted on a
  // pure zoom change so the backend keeps its own (edge-following) pan.
  onViewChange?: (zoom: number, pan?: number) => void;

  // Notified whenever the zoom level changes for ANY reason (wheel, pinch, or
  // a programmatic setZoom from the slider). The UI uses this to keep the zoom
  // slider in sync with mouse-wheel zooming -- they are the same single zoom.
  onZoomChange?: (zoom: number) => void;

  // Min / max zoom, shared by the wheel, pinch and the slider so all three map
  // to the exact same range.
  static readonly MIN_ZOOM = 1;
  static readonly MAX_ZOOM = 64;

  constructor(private readonly canvas: HTMLCanvasElement, fftSize: number, historyRows = 1080) {
    const gl = canvas.getContext('webgl2', {
      alpha: false, antialias: false, depth: false, stencil: false,
      powerPreference: 'high-performance',
    });
    if (!gl) throw new Error('WebGL 2.0 is required for the SDR waterfall');

    this.gl = gl;
    this.sourceWidth = fftSize;
    this.textureWidth = Math.min(fftSize, Number(gl.getParameter(gl.MAX_TEXTURE_SIZE)));
    this.height = historyRows;
    this.rowBuffer = new Uint8Array(this.textureWidth);
    this.program = buildProgram(gl);
    this.vao = this.createQuad();
    this.wfTex = this.createWaterfallTexture();
    this.lutTex = this.createLutTexture('spectrum');
    this.bindStaticUniforms();
    this.setupZoomPan();
  }

  private setupZoomPan(): void {
    const canvas = this.canvas;

    // Mouse wheel: zoom in/out around the tuned frequency.
    canvas.addEventListener('wheel', (e) => {
      e.preventDefault();
      this.setZoom(this.zoom * (e.deltaY < 0 ? 1.25 : 0.8));
    }, { passive: false });

    // Touch pinch zoom, also centred.
    let lastTouchDist = 0;
    canvas.addEventListener('touchstart', (e) => {
      if (e.touches.length === 2) {
        const dx = e.touches[0].clientX - e.touches[1].clientX;
        lastTouchDist = Math.abs(dx);
      }
    }, { passive: true });
    canvas.addEventListener('touchmove', (e) => {
      if (e.touches.length === 2) {
        const dx = e.touches[0].clientX - e.touches[1].clientX;
        const dist = Math.abs(dx);
        if (lastTouchDist > 0) this.setZoom(this.zoom * (dist / lastTouchDist));
        lastTouchDist = dist;
      }
    }, { passive: true });
  }

  // Single source of truth for changing zoom. Called by the mouse wheel, touch
  // pinch AND the UI zoom slider -- they are all the same zoom. Clamps to the
  // shared range, updates the backend view (zoom only, pan kept by backend) and
  // notifies the UI via onZoomChange so the slider tracks wheel/pinch zooming.
  //
  // Zoom narrows the visible span. The zoom window keeps its current centre
  // (pan) -- the backend owns pan and only moves it when the tuned signal
  // reaches the window edge -- so zooming in does NOT recentre on the tuned
  // signal: it stays put where you clicked.
  setZoom(next: number): void {
    const z = Math.max(WaterfallRenderer.MIN_ZOOM,
                       Math.min(WaterfallRenderer.MAX_ZOOM, next));
    if (z === this.zoom) return;
    this.zoom = z;
    this.onViewChange?.(this.zoom);
    this.onZoomChange?.(this.zoom);
  }

  // Live drag offset in screen-fraction (-1..1): how far the visible slice is
  // shifted horizontally while the user drags. Set during a drag, reset to 0 on
  // release (when the backend delivers a re-centred slice). main.ts owns the
  // drag gesture because it knows the on-screen slice's absolute frequency.
  setDragOffset(offset: number): void {
    this.dragOffset = offset;
  }

  // Current live drag offset (screen-fraction). main.ts reads this so the
  // frequency ruler + band plan can shift in lockstep with the dragged image
  // (otherwise the strip stays put while the waterfall slides -> they disagree).
  get currentDragOffset(): number {
    return this.dragOffset;
  }

  pushRow(bins: Float32Array): void {
    if (bins.length !== this.sourceWidth) return;
    const gl = this.gl;
    for (let i = 0; i < this.textureWidth; i++) {
      const start = Math.floor((i * this.sourceWidth) / this.textureWidth);
      const end = Math.max(start + 1, Math.floor(((i + 1) * this.sourceWidth) / this.textureWidth));
      let db = Number.NEGATIVE_INFINITY;
      for (let j = start; j < end; j++) db = Math.max(db, bins[j]);
      const encoded = ((db - ENCODE_MIN_DB) / ENCODE_SPAN_DB) * 255;
      this.rowBuffer[i] = Math.min(255, Math.max(0, Math.round(encoded)));
    }
    gl.bindTexture(gl.TEXTURE_2D, this.wfTex);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, this.writeRow % this.height,
      this.textureWidth, 1, gl.RED, gl.UNSIGNED_BYTE, this.rowBuffer);
    this.writeRow += 1;
  }

  draw(dbMin: number, dbMax: number): void {
    const gl = this.gl;
    this.resize();
    gl.viewport(0, 0, gl.drawingBufferWidth, gl.drawingBufferHeight);
    gl.useProgram(this.program);
    gl.bindVertexArray(this.vao);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.wfTex);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.lutTex);

    gl.uniform1f(gl.getUniformLocation(this.program, 'u_min'), dbMin);
    gl.uniform1f(gl.getUniformLocation(this.program, 'u_max'), dbMax);
    gl.uniform1f(gl.getUniformLocation(this.program, 'u_head'), (this.writeRow % this.height) / this.height);
    // Server-side zoom: the incoming texture is already exactly the visible
    // slice, so the shader draws it 1:1 (no GPU zoom/pan window).
    gl.uniform1f(gl.getUniformLocation(this.program, 'u_zoom'), 1.0);
    gl.uniform1f(gl.getUniformLocation(this.program, 'u_pan'), 0.5);
    // While dragging, slide the slice horizontally so the image tracks the mouse.
    gl.uniform1f(gl.getUniformLocation(this.program, 'u_drag'), this.dragOffset);
    // Viewport width in device pixels: lets the shader peak-decimate exactly the
    // FFT bins that fall under each pixel (sharp narrow signals, no smear).
    gl.uniform1f(gl.getUniformLocation(this.program, 'u_px'), gl.drawingBufferWidth);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }

  setColorMap(name: ColorMapName): void {
    const gl = this.gl;
    gl.deleteTexture(this.lutTex);
    this.lutTex = this.createLutTexture(name);
  }

  destroy(): void {
    const gl = this.gl;
    gl.deleteTexture(this.wfTex);
    gl.deleteTexture(this.lutTex);
    gl.deleteVertexArray(this.vao);
    gl.deleteProgram(this.program);
  }

  private resize(): void {
    const ratio = window.devicePixelRatio || 1;
    const width = Math.max(1, Math.floor(this.canvas.clientWidth * ratio));
    const height = Math.max(1, Math.floor(this.canvas.clientHeight * ratio));
    if (this.canvas.width !== width || this.canvas.height !== height) {
      this.canvas.width = width;
      this.canvas.height = height;
    }
  }

  private bindStaticUniforms(): void {
    const gl = this.gl;
    gl.useProgram(this.program);
    gl.uniform1i(gl.getUniformLocation(this.program, 'u_wf'), 0);
    gl.uniform1i(gl.getUniformLocation(this.program, 'u_lut'), 1);
  }

  private createQuad(): WebGLVertexArrayObject {
    const gl = this.gl;
    const vao = gl.createVertexArray();
    const buffer = gl.createBuffer();
    if (!vao || !buffer) throw new Error('failed to create WebGL quad');
    gl.bindVertexArray(vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
    const location = gl.getAttribLocation(this.program, 'a_pos');
    gl.enableVertexAttribArray(location);
    gl.vertexAttribPointer(location, 2, gl.FLOAT, false, 0, 0);
    return vao;
  }

  private createWaterfallTexture(): WebGLTexture {
    const gl = this.gl;
    const texture = gl.createTexture();
    if (!texture) throw new Error('failed to create waterfall texture');
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texStorage2D(gl.TEXTURE_2D, 1, gl.R8, this.textureWidth, this.height);
    // NEAREST: the fragment shader does its own bilinear-in-frequency /
    // nearest-in-time sampling via texelFetch (which ignores GL filtering),
    // so we don't want a second round of hardware filtering on top.
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    const empty = new Uint8Array(this.textureWidth * this.height);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, this.textureWidth, this.height, gl.RED, gl.UNSIGNED_BYTE, empty);
    return texture;
  }

  private createLutTexture(name: ColorMapName): WebGLTexture {
    const gl = this.gl;
    const texture = gl.createTexture();
    if (!texture) throw new Error('failed to create LUT texture');
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 256, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, makeColorMap(name));
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    return texture;
  }
}
