from fastapi import FastAPI, File, UploadFile
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
import soundfile as sf
import io

# Load model once at startup
MODEL_NAME = "facebook/wav2vec2-large-960h"  # English-only
processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME)

app = FastAPI()

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    # Read audio file
    audio_bytes = await file.read()
    speech, sample_rate = sf.read(io.BytesIO(audio_bytes))

    # Prepare input for model
    inputs = processor(speech, sampling_rate=sample_rate, return_tensors="pt", padding=True)

    # Run model inference
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_ids = torch.argmax(logits, dim=-1)

    # Decode into text
    transcription = processor.batch_decode(predicted_ids)[0]
    return {"text": transcription}
