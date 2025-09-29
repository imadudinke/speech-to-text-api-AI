# app.py

from fastapi import FastAPI, File, UploadFile, HTTPException
from faster_whisper import WhisperModel
import io

# Initialize FastAPI app
app = FastAPI()

# --- Model Loading ---
# Use the tiny English-only model, which is very lightweight.
# It uses approximately 75MB of RAM with int8 precision.
MODEL_NAME = "tiny.en" 
device = "cpu"  # Force CPU usage to avoid CUDA memory issues
compute_type = "int8"  # Use 8-bit integer quantization for minimal memory usage

try:
    # Load the Whisper model once at startup for all requests
    model = WhisperModel(MODEL_NAME, device=device, compute_type=compute_type)
    print(f"Loaded model '{MODEL_NAME}' with '{compute_type}' on '{device}'.")
except Exception as e:
    raise RuntimeError(f"Failed to load Whisper model: {e}")


@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Transcribes an audio file uploaded via a POST request.
    """
    # Check for correct audio format
    if not file.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="Invalid file format. Please upload an audio file.")

    try:
        # Read audio file into an in-memory buffer
        audio_bytes = await file.read()
        audio_stream = io.BytesIO(audio_bytes)

        # Transcribe the audio using the faster-whisper model
        # The model automatically handles audio chunking, making it robust for longer files
        segments, _ = model.transcribe(audio_stream)

        # Collect the transcription text from all segments
        transcription = " ".join([segment.text for segment in segments])
        
        return {"text": transcription}
    
    except Exception as e:
        # Catch and handle potential transcription errors gracefully
        # This can be for unsupported audio formats, corrupted files, etc.
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")

@app.get("/")
async def root():
    return {"message": "Faster Whisper API is running. Send a POST request to /transcribe to transcribe audio."}

