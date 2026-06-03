import { ColorMapName, makeColorMap } from './colormap';

const VERT_SRC = `#version 300 es
  in vec2 a_pos;
  out vec2 v_uv;

  void main() {
    v_uv = a_pos * 0.5 + 0.5;
    gl_Position = vec4(a_pos, 0.0, 1.0);
  }
`;

const FRAG_SRC = `#version 300 es
  precision highp float;

  uniform sampler2D u_wf;
  uniform sampler2D u_lut;
  uniform float u_min;
  uniform float u_max;
  uniform float u_head;

  in vec2 v_uv;
  out vec4 fragColor;

  void main() {
    float y = mod(u_head - v_uv.y + 1.0, 1.0);
    float db = texture(u_wf, vec2(v_uv.x, y)).r * 200.0 - 160.0;
    float t = clamp((db - u_min) / max(0.001, u_max - u_min), 0.0, 1.0);
    fragColor = texture(u_lut, vec2(t, 0.5));
  }
`;

const ENCODE_MIN_DB = -160;
const ENCODE_SPAN_DB = 200;

function compileShader(gl: WebGL2RenderingContext, type: number, source: string): WebGLShader {
  const shader = gl.createShader(type);
  if (!shader) {
    throw new Error('failed to create WebGL shader');
  }
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
  if (!program) {
    throw new Error('failed to create WebGL program');
  }
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

  constructor(private readonly canvas: HTMLCanvasElement, fftSize: number, historyRows = 640) {
    const gl = canvas.getContext('webgl2', {
      alpha: false,
      antialias: false,
      depth: false,
      stencil: false,
      powerPreference: 'high-performance',
    });
    if (!gl) {
      throw new Error('WebGL 2.0 is required for the SDR waterfall');
    }

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
  }

  pushRow(bins: Float32Array): void {
    if (bins.length !== this.sourceWidth) {
      return;
    }

    const gl = this.gl;
    for (let i = 0; i < this.textureWidth; i++) {
      const start = Math.floor((i * this.sourceWidth) / this.textureWidth);
      const end = Math.max(start + 1, Math.floor(((i + 1) * this.sourceWidth) / this.textureWidth));
      let db = Number.NEGATIVE_INFINITY;
      for (let j = start; j < end; j++) {
        db = Math.max(db, bins[j]);
      }
      const encoded = ((db - ENCODE_MIN_DB) / ENCODE_SPAN_DB) * 255;
      this.rowBuffer[i] = Math.min(255, Math.max(0, Math.round(encoded)));
    }

    gl.bindTexture(gl.TEXTURE_2D, this.wfTex);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.texSubImage2D(
      gl.TEXTURE_2D,
      0,
      0,
      this.writeRow % this.height,
      this.textureWidth,
      1,
      gl.RED,
      gl.UNSIGNED_BYTE,
      this.rowBuffer,
    );
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
    if (!vao || !buffer) {
      throw new Error('failed to create WebGL quad');
    }

    gl.bindVertexArray(vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(
      gl.ARRAY_BUFFER,
      new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]),
      gl.STATIC_DRAW,
    );

    const location = gl.getAttribLocation(this.program, 'a_pos');
    gl.enableVertexAttribArray(location);
    gl.vertexAttribPointer(location, 2, gl.FLOAT, false, 0, 0);
    return vao;
  }

  private createWaterfallTexture(): WebGLTexture {
    const gl = this.gl;
    const texture = gl.createTexture();
    if (!texture) {
      throw new Error('failed to create waterfall texture');
    }

    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texStorage2D(gl.TEXTURE_2D, 1, gl.R8, this.textureWidth, this.height);
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
    if (!texture) {
      throw new Error('failed to create LUT texture');
    }

    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 256, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, makeColorMap(name));
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    return texture;
  }
}
