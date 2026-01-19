"""ASR Service using FastAPI with Sarvam AI."""

import base64
import io
import logging
from contextlib import asynccontextmanager
from typing import Optional

import requests
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SARVAM_API_URL = "https://api.sarvam.ai/speech-to-text"


class TranscriptionResponse(BaseModel):
    text: str
    language: str
    duration_seconds: Optional[float] = None


class Base64AudioRequest(BaseModel):
    audio_base64: str
    language_code: Optional[str] = None


class ASRService:
    """ASR using Sarvam AI's speech-to-text API."""

    def __init__(self):
        self.api_key = config.SARVAM_API_KEY
        self._language = "hi"
        if not self.api_key:
            raise ValueError("SARVAM_API_KEY not set in config")

    @property
    def language(self) -> str:
        return self._language

    def load(self) -> None:
        logger.info("Sarvam AI ASR ready (API-based, no model loading needed)")

    def transcribe(self, audio_file, language_code: str = None) -> str:
        """Transcribe audio using Sarvam API."""
        headers = {"api-subscription-key": self.api_key}
        data = {}
        if language_code:
            data["language_code"] = language_code

        response = requests.post(
            SARVAM_API_URL,
            headers=headers,
            files={"file": audio_file},
            data=data,
            timeout=30
        )
        response.raise_for_status()
        result = response.json()

        self._language = result.get("language_code", "hi-IN").split("-")[0]
        return result.get("transcript", "").strip()


# App setup
asr: Optional[ASRService] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global asr
    asr = ASRService()
    asr.load()
    yield


def get_asr() -> ASRService:
    if not asr:
        raise HTTPException(503, "Service not ready")
    return asr


app = FastAPI(title="ASR Service - Sarvam AI", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/health")
async def health():
    return {"status": "healthy" if asr else "loading", "provider": "Sarvam AI"}


@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe(
    file: UploadFile = File(...),
    language: str = Query(None),
    svc: ASRService = Depends(get_asr)
):
    """Transcribe audio file using Sarvam AI."""
    try:
        audio_bytes = await file.read()
        audio_buffer = io.BytesIO(audio_bytes)
        audio_file = (file.filename or "audio.wav", audio_buffer, file.content_type or "audio/wav")

        text = svc.transcribe(audio_file, language_code=language)
        duration = len(audio_bytes) / 16000  # rough estimate

        return TranscriptionResponse(
            text=text,
            language=svc.language,
            duration_seconds=round(duration, 2)
        )
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        raise HTTPException(500, f"Transcription failed: {str(e)}")


@app.post("/transcribe/base64", response_model=TranscriptionResponse)
async def transcribe_b64(req: Base64AudioRequest, svc: ASRService = Depends(get_asr)):
    """Transcribe base64 encoded audio using Sarvam AI."""
    audio_bytes = base64.b64decode(req.audio_base64)
    audio_file = ("audio.wav", io.BytesIO(audio_bytes), "audio/wav")

    text = svc.transcribe(audio_file, language_code=req.language_code)
    duration = len(audio_bytes) / 16000  # rough estimate

    return TranscriptionResponse(
        text=text,
        language=svc.language,
        duration_seconds=round(duration, 2)
    )


if __name__ == "__main__":
    import uvicorn
    print(f"\nASR: http://{config.ASR_HOST}:{config.ASR_PORT}/docs\n")
    uvicorn.run("task3_asr_service:app", host=config.ASR_HOST, port=config.ASR_PORT, reload=True)
