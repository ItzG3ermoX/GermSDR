import { parseWaterfallFrame, WaterfallFrame } from './protocol';

function websocketUrl(path: string): string {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  return `${proto}://${location.host}${path}`;
}

export interface WaterfallClient {
  close: () => void;
  // Drive server-side zoom. Pan is optional: omit it to change only the zoom
  // level and let the backend keep its own (edge-following) pan position.
  setView: (zoom: number, pan?: number) => void;
  // Drag-to-pan: move the view centre to an ABSOLUTE frequency (Hz). The backend
  // converts it to the captured-band fraction (it owns the band centre/rate).
  panToHz: (hz: number) => void;
}

export function connectWaterfall(
  onFrame: (frame: WaterfallFrame) => void,
  onState: (state: 'connecting' | 'open' | 'closed') => void,
): WaterfallClient {
  let closed = false;
  let ws: WebSocket | undefined;
  let retry: number | undefined;
  let lastView: { zoom: number; pan?: number } = { zoom: 1 };

  const connect = () => {
    onState('connecting');
    ws = new WebSocket(websocketUrl('/ws/waterfall'));
    ws.binaryType = 'arraybuffer';

    ws.onopen = () => {
      onState('open');
      // Re-assert the current view on (re)connect so the server resumes the zoom.
      ws?.send(JSON.stringify(lastView));
    };
    ws.onmessage = (event) => {
      try {
        onFrame(parseWaterfallFrame(event.data as ArrayBuffer));
      } catch (e) {
        console.warn('waterfall frame handler error:', e);
      }
    };
    ws.onclose = () => {
      onState('closed');
      if (!closed) {
        retry = window.setTimeout(connect, 1200);
      }
    };
    ws.onerror = () => ws?.close();
  };

  connect();

  return {
    close: () => {
      closed = true;
      if (retry) {
        window.clearTimeout(retry);
      }
      ws?.close();
    },
    setView: (zoom: number, pan?: number) => {
      // When pan is omitted, change only zoom and keep the last known pan so we
      // re-assert a consistent view on reconnect; the message itself carries
      // only the fields we want the server to act on.
      lastView = pan === undefined ? { zoom } : { zoom, pan };
      if (ws && ws.readyState === WebSocket.OPEN) {
        const msg = pan === undefined ? { zoom } : { zoom, pan };
        ws.send(JSON.stringify(msg));
      }
    },
    panToHz: (hz: number) => {
      // The backend owns the pan fraction after an absolute-frequency drag.
      // Do not overwrite it with a stale value if the socket reconnects.
      delete lastView.pan;
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ pan_hz: hz }));
      }
    },
  };
}

export interface AudioClient {
  ctx: AudioContext;
  node: AudioWorkletNode;
  ws: WebSocket;
  close: () => void;
}

export async function connectAudio(): Promise<AudioClient> {
  const ctx = new AudioContext({ sampleRate: 48_000, latencyHint: 'interactive' });

  // Warn if the browser created a different rate than asked — the backend
  // always outputs 48 kHz PCM, so a rate mismatch would play at wrong pitch.
  if (ctx.sampleRate !== 48_000) {
    await ctx.close();
    throw new Error(`Audio output requires 48000 Hz; browser selected ${ctx.sampleRate} Hz.`);
  }

  await ctx.audioWorklet.addModule('/audio-processor.js');
  const node = new AudioWorkletNode(ctx, 'sdr-audio', {
    numberOfOutputs: 1,
    outputChannelCount: [2],
  });
  node.connect(ctx.destination);

  let closed = false;
  let ws: WebSocket | undefined;
  let reconnectTimer: number | undefined;

  function createWs(): void {
    // Clear any stale reconnect timer so rapid close/reopen cycles
    // don't stack multiple reconnect attempts.
    if (reconnectTimer !== undefined) {
      window.clearTimeout(reconnectTimer);
      reconnectTimer = undefined;
    }
    // Close the previous WS if it's still lingering (e.g., half-open).
    if (ws && ws.readyState !== WebSocket.CLOSED && ws.readyState !== WebSocket.CLOSING) {
      ws.close();
    }

    const w = new WebSocket(websocketUrl('/ws/audio'));
    w.binaryType = 'arraybuffer';
    w.onmessage = (event) => {
      const pcm = event.data as ArrayBuffer;
      node.port.postMessage({ pcm }, [pcm]);
    };
    w.onclose = () => {
      if (!closed) {
        reconnectTimer = window.setTimeout(() => {
          reconnectTimer = undefined;
          if (!closed) createWs();
        }, 1200);
      }
    };
    w.onerror = () => w.close();
    ws = w;
  }

  createWs();

  return {
    ctx,
    node,
    ws: ws!,
    close: () => {
      closed = true;
      if (reconnectTimer !== undefined) {
        window.clearTimeout(reconnectTimer);
        reconnectTimer = undefined;
      }
      ws?.close();
      node.disconnect();
      void ctx.close();
    },
  };
}
