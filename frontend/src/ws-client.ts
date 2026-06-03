import { parseWaterfallFrame, WaterfallFrame } from './protocol';

function websocketUrl(path: string): string {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  return `${proto}://${location.host}${path}`;
}

export interface WaterfallClient {
  close: () => void;
}

export function connectWaterfall(
  onFrame: (frame: WaterfallFrame) => void,
  onState: (state: 'connecting' | 'open' | 'closed') => void,
): WaterfallClient {
  let closed = false;
  let ws: WebSocket | undefined;
  let retry: number | undefined;

  const connect = () => {
    onState('connecting');
    ws = new WebSocket(websocketUrl('/ws/waterfall'));
    ws.binaryType = 'arraybuffer';

    ws.onopen = () => onState('open');
    ws.onmessage = (event) => onFrame(parseWaterfallFrame(event.data as ArrayBuffer));
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
  await ctx.audioWorklet.addModule('/audio-processor.js');
  const node = new AudioWorkletNode(ctx, 'sdr-audio', {
    numberOfOutputs: 1,
    outputChannelCount: [2],
  });
  node.connect(ctx.destination);

  const ws = new WebSocket(websocketUrl('/ws/audio'));
  ws.binaryType = 'arraybuffer';
  ws.onmessage = (event) => {
    const pcm = event.data as ArrayBuffer;
    node.port.postMessage({ pcm }, [pcm]);
  };

  return {
    ctx,
    node,
    ws,
    close: () => {
      ws.close();
      node.disconnect();
      void ctx.close();
    },
  };
}

