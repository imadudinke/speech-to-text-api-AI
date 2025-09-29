from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from faster_whisper import WhisperModel
import io
import asyncio

app = FastAPI()

MODEL_NAME = "tiny.en"
device = "cpu"
compute_type = "int8"

# Load the model once at startup
model = WhisperModel(MODEL_NAME, device=device, compute_type=compute_type)
print(f"Loaded model '{MODEL_NAME}' with '{compute_type}' on '{device}'.")

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    if not file.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="Invalid file format. Please upload an audio file.")
    
    audio_bytes = await file.read()
    audio_stream = io.BytesIO(audio_bytes)

    async def generate_transcription():
        # Use beam_size=1 for faster inference
        segments, _ = model.transcribe(audio_stream, beam_size=1)

        # Yield segments as they are generated
        for segment in segments:
            yield segment.text + " "
            # Small async sleep to prevent blocking FastAPI
            await asyncio.sleep(0)

    return StreamingResponse(generate_transcription(), media_type="text/plain")
