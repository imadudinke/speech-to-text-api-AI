# app.py
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse
from faster_whisper import WhisperModel
import numpy as np

app = FastAPI(title="Realtime Whisper (WebSocket)")

# -------------------
# Config
# -------------------
MODEL_NAME = "tiny.en"        # keep tiny.en for CPU usage; change if you have GPU
DEVICE = "cpu"               # "cpu" or "cuda"
COMPUTE_TYPE = "int8"        # int8 for smallest memory footprint if supported

SAMPLE_RATE = 16000          # audio sample rate we expect from client
BYTES_PER_SAMPLE = 2         # int16 = 2 bytes
CHUNK_SECONDS = 2.0          # seconds per chunk to process
CHUNK_BYTES = int(SAMPLE_RATE * BYTES_PER_SAMPLE * CHUNK_SECONDS)

# -------------------
# Load model once
# -------------------
print("Loading model... (this happens once at startup)")
model = WhisperModel(MODEL_NAME, device=DEVICE, compute_type=COMPUTE_TYPE)
print(f"Loaded model {MODEL_NAME} on {DEVICE} / {COMPUTE_TYPE}")

# -------------------
# Simple root + HTML client (optional)
# -------------------
@app.get("/", response_class=HTMLResponse)
async def index():
    # Very small front-end to test microphone streaming (JS included).
    # When deployed under https, change ws:// -> wss:// in client.
    return """
<!doctype html>
<html>
  <head><meta charset="utf-8"><title>Realtime STT (WebSocket)</title></head>
  <body>
    <h3>Realtime STT test</h3>
    <button onclick="start()">Start</button>
    <button onclick="stop()">Stop</button>
    <pre id="out"></pre>
<script>
let ws;
let proc;
let audioCtx;
let source;
let recordNode;

function floatTo16BitPCM(float32Array) {
  const l = float32Array.length;
  const buffer = new ArrayBuffer(l * 2);
  const view = new DataView(buffer);
  let offset = 0;
  for (let i = 0; i < l; i++, offset += 2) {
    let s = Math.max(-1, Math.min(1, float32Array[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
  }
  return buffer;
}

// downsample & convert to 16k (if input sampleRate != 16000)
function downsampleBuffer(buffer, sampleRate, outSampleRate) {
  if (outSampleRate === sampleRate) return buffer;
  const sampleRateRatio = sampleRate / outSampleRate;
  const newLength = Math.round(buffer.length / sampleRateRatio);
  const result = new Float32Array(newLength);
  let offsetResult = 0;
  let offsetBuffer = 0;
  while (offsetResult < result.length) {
    const nextOffsetBuffer = Math.round((offsetResult + 1) * sampleRateRatio);
    // average samples between offsetBuffer & nextOffsetBuffer
    let accum = 0, count = 0;
    for (let i = offsetBuffer; i < nextOffsetBuffer && i < buffer.length; i++) {
      accum += buffer[i];
      count++;
    }
    result[offsetResult] = accum / Math.max(count, 1);
    offsetResult++;
    offsetBuffer = nextOffsetBuffer;
  }
  return result;
}

async function start() {
  ws = new WebSocket((location.protocol === 'https:' ? 'wss' : 'ws') + '://' + location.host + '/ws');
  ws.binaryType = 'arraybuffer';
  ws.onopen = () => console.log('ws open');
  ws.onmessage = (ev) => {
    document.getElementById('out').textContent += ev.data;
  };
  ws.onclose = () => console.log('ws closed');

  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  source = audioCtx.createMediaStreamSource(stream);

  // Using ScriptProcessorNode (widely supported). Buffer size 4096 chosen.
  recordNode = audioCtx.createScriptProcessor(4096, 1, 1);
  recordNode.onaudioprocess = function(e) {
    let input = e.inputBuffer.getChannelData(0);
    // downsample to 16k if needed
    const down = downsampleBuffer(input, audioCtx.sampleRate, 16000);
    const pcm16 = floatTo16BitPCM(down);
    if (ws && ws.readyState === 1) {
      ws.send(pcm16);
    }
  };
  source.connect(recordNode);
  recordNode.connect(audioCtx.destination);
}

function stop() {
  if (recordNode) {
    recordNode.disconnect();
    recordNode = null;
  }
  if (source) {
    source.disconnect();
    source = null;
  }
  if (audioCtx) {
    audioCtx.close();
    audioCtx = null;
  }
  if (ws) {
    ws.close();
    ws = null;
  }
}
</script>
  </body>
</html>
"""

# -------------------
# WebSocket endpoint: client sends raw PCM16LE 16000Hz bytes
# -------------------
from fastapi import WebSocketDisconnect

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    buffer = bytearray()
    loop = asyncio.get_running_loop()
    # track how many seconds of audio we've processed so far (used for absolute timestamps)
    processed_seconds = 0.0
    last_sent_end = 0.0

    try:
        while True:
            # receive bytes from client. client sends raw PCM int16 frames
            data = await websocket.receive_bytes()
            buffer.extend(data)

            # process as many full chunks as available
            while len(buffer) >= CHUNK_BYTES:
                chunk = bytes(buffer[:CHUNK_BYTES])
                del buffer[:CHUNK_BYTES]

                # convert bytes -> numpy float32 normalized (-1..1)
                audio_int16 = np.frombuffer(chunk, dtype=np.int16)
                audio_float = audio_int16.astype(np.float32) / 32768.0

                # heavy CPU work must run in a threadpool to avoid blocking event loop
                def transcribe_chunk(arr):
                    # arr is float32 mono numpy array at SAMPLE_RATE
                    segments, _ = model.transcribe(arr, beam_size=1, word_timestamps=False)
                    return segments

                segments = await loop.run_in_executor(None, transcribe_chunk, audio_float)

                # collect only new segments (by absolute time)
                send_texts = []
                max_end = last_sent_end
                for seg in segments:
                    # seg.start and seg.end are relative to this chunk (seconds)
                    abs_start = seg.start + processed_seconds
                    abs_end = seg.end + processed_seconds
                    if abs_end > last_sent_end + 1e-3:
                        send_texts.append(seg.text.strip())
                        if abs_end > max_end:
                            max_end = abs_end

                if send_texts:
                    # send concatenated new text to client
                    await websocket.send_text(" ".join(send_texts))
                    last_sent_end = max_end

                processed_seconds += CHUNK_SECONDS

    except WebSocketDisconnect:
        # client closed
        return
    except Exception as e:
        try:
            await websocket.send_text(f"[ERROR] {e}")
        except:
            pass
        return
