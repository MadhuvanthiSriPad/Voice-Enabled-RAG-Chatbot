"""ASR Service using FastAPI with AI4Bharat IndicWav2Vec model."""

import base64
import io
import logging
import os
import tempfile
from contextlib import asynccontextmanager
from typing import Optional

import librosa
import numpy as np
import soundfile as sf
import torch
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000


class TranscriptionResponse(BaseModel):
    text: str
    language: str
    duration_seconds: Optional[float] = None


class Base64AudioRequest(BaseModel):
    audio_base64: str


class ASRService:
    """Loads and runs ASR model."""

    def __init__(self, model_name: str = None):
        self.model_name = model_name or config.ASR_MODEL_NAME
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.processor = None

    @property
    def language(self) -> str:
        for lang, code in {"hindi": "hi", "bengali": "bn", "tamil": "ta", "telugu": "te"}.items():
            if lang in self.model_name.lower():
                return code
        return "en"

    def load(self) -> None:
        logger.info(f"Loading: {self.model_name}")
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
        self.processor = Wav2Vec2Processor.from_pretrained(self.model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(self.model_name)
        self.model.to(self.device).eval()
        logger.info("Model loaded")

    def transcribe(self, audio: np.ndarray) -> str:
        inputs = self.processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = self.model(inputs.input_values.to(self.device)).logits
        return self.processor.batch_decode(torch.argmax(logits, dim=-1))[0]


def load_audio(data: bytes) -> np.ndarray:
    try:
        audio, sr = sf.read(io.BytesIO(data))
        return librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE) if sr != SAMPLE_RATE else audio
    except Exception:
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            f.write(data)
            path = f.name
        try:
            audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
            return audio
        finally:
            os.unlink(path)


# App setup
asr: Optional[ASRService] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global asr
    asr = ASRService()
    asr.load()
    yield


def get_asr() -> ASRService:
    if not asr or not asr.model:
        raise HTTPException(503, "Service not ready")
    return asr


app = FastAPI(title="ASR Service", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/health")
async def health():
    return {"status": "healthy" if asr and asr.model else "loading", "model": asr.model_name if asr else None}


@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe(file: UploadFile = File(...), svc: ASRService = Depends(get_asr)):
    audio = load_audio(await file.read())
    return TranscriptionResponse(text=svc.transcribe(audio), language=svc.language, duration_seconds=round(len(audio) / SAMPLE_RATE, 2))


@app.post("/transcribe/base64", response_model=TranscriptionResponse)
async def transcribe_b64(req: Base64AudioRequest, svc: ASRService = Depends(get_asr)):
    audio = load_audio(base64.b64decode(req.audio_base64))
    return TranscriptionResponse(text=svc.transcribe(audio), language=svc.language, duration_seconds=round(len(audio) / SAMPLE_RATE, 2))


if __name__ == "__main__":
    import uvicorn
    print(f"\nASR: http://{config.ASR_HOST}:{config.ASR_PORT}/docs\n")
    uvicorn.run("task3_asr_service:app", host=config.ASR_HOST, port=config.ASR_PORT, reload=True)
