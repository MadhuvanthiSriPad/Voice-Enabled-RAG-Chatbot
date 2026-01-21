"""ASR Service using FastAPI with AI4Bharat IndicConformer Model (Transformers)."""

import base64
import io
import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

import torch
import torchaudio
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoModel

from config import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Model configuration
MODEL_NAME = "ai4bharat/indic-conformer-600m-multilingual"
TARGET_SAMPLE_RATE = 16000
DECODER_TYPE = "rnnt"  # Use RNNT for better quality (can also use "ctc")

# Validation constants
MAX_FILE_SIZE_MB = 100
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
SUPPORTED_AUDIO_FORMATS = ['.wav', '.flac', '.mp3', '.m4a', '.ogg', '.webm']

# Supported languages (22 Indian languages)
SUPPORTED_LANGUAGES = [
    "as", "bn", "brx", "doi", "gu", "hi", "kn", "kok", "ks", "mai",
    "ml", "mni", "mr", "ne", "or", "pa", "sa", "sat", "sd", "ta", "te", "ur"
]


class TranscriptionResponse(BaseModel):
    text: str
    language: str
    duration_seconds: Optional[float] = None


class Base64AudioRequest(BaseModel):
    audio_base64: str
    language_code: str  # REQUIRED


class ASRService:
    """ASR using AI4Bharat's IndicConformer model with Transformers."""

    def __init__(self):
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._language = None
        logger.info(f"Initializing ASR service on device: {self.device}")

    @property
    def language(self) -> str:
        return self._language

    def load(self) -> None:
        """Load the IndicConformer model using Transformers."""
        try:
            print(f"Loading IndicConformer model: {MODEL_NAME}", flush=True)
            logger.info(f"Loading IndicConformer model: {MODEL_NAME}")

            # Load model using Transformers with trust_remote_code
            print("Downloading/loading model... This may take several minutes on first run.", flush=True)
            self.model = AutoModel.from_pretrained(
                MODEL_NAME,
                trust_remote_code=True
            )
            print("Model loaded successfully!", flush=True)

            # Move to device
            self.model = self.model.to(self.device)
            self.model.eval()

            logger.info(f"IndicConformer model loaded successfully (decoder: {DECODER_TYPE})")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def _prepare_audio(self, audio_bytes: bytes) -> tuple[torch.Tensor, float]:
        """Load audio from bytes, convert to mono, and resample to 16kHz."""
        try:
            # Load audio from bytes
            audio_buffer = io.BytesIO(audio_bytes)
            wav, sr = torchaudio.load(audio_buffer)

            # Calculate accurate duration from original audio
            duration = wav.shape[1] / sr

            # Convert stereo to mono by averaging channels
            if wav.shape[0] > 1:
                wav = torch.mean(wav, dim=0, keepdim=True)

            # Resample if necessary
            if sr != TARGET_SAMPLE_RATE:
                logger.info(f"Resampling audio from {sr}Hz to {TARGET_SAMPLE_RATE}Hz")
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sr,
                    new_freq=TARGET_SAMPLE_RATE
                )
                wav = resampler(wav)

            return wav, duration

        except Exception as e:
            logger.error(f"Audio processing error: {e}")
            raise ValueError(f"Failed to process audio: {str(e)}")

    def transcribe(self, audio_bytes: bytes, language_code: str) -> tuple[str, float]:
        """Transcribe audio bytes using IndicConformer model for the specified language."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Validate language code is provided
        if not language_code:
            raise ValueError("language_code is required")

        # Convert language code from "hi-IN" format to "hi" format if needed
        lang_code = language_code.split('-')[0] if '-' in language_code else language_code

        if lang_code not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported language: {lang_code}. Supported: {', '.join(SUPPORTED_LANGUAGES)}")

        self._language = lang_code

        logger.info(f"Transcribing audio in language: {lang_code}")

        try:
            # Prepare audio and get tensor
            wav, duration = self._prepare_audio(audio_bytes)

            # Move tensor to device
            wav = wav.to(self.device)

            # Perform ASR using the model
            # Model signature: model(wav_tensor, language_code, decoder_type)
            with torch.no_grad():
                transcription = self.model(wav, lang_code, DECODER_TYPE)

            logger.info(f"Transcription successful: {transcription[:100]}...")
            return transcription.strip(), duration

        except ValueError as e:
            # Re-raise validation errors
            raise
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            raise RuntimeError(f"Transcription failed: {str(e)}")


# App setup
asr: Optional[ASRService] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    global asr
    print("=" * 60, flush=True)
    print("Starting ASR service...", flush=True)
    print("=" * 60, flush=True)
    logger.info("Starting ASR service...")
    asr = ASRService()
    asr.load()
    print("ASR service ready!", flush=True)
    logger.info("ASR service ready")
    yield
    logger.info("Shutting down ASR service")


def get_asr() -> ASRService:
    """Dependency to get ASR service instance."""
    if not asr:
        raise HTTPException(503, "Service not ready")
    return asr


def validate_audio_file(file: UploadFile) -> None:
    """
    Validate uploaded audio file.

    Raises HTTPException with 400 status for invalid input.
    """
    # Check if filename exists
    if not file.filename:
        raise HTTPException(400, "Filename is required")

    # Check file extension
    file_ext = '.' + file.filename.split('.')[-1].lower() if '.' in file.filename else ''
    if file_ext not in SUPPORTED_AUDIO_FORMATS:
        raise HTTPException(
            400,
            f"Unsupported audio format: {file_ext}. Supported formats: {', '.join(SUPPORTED_AUDIO_FORMATS)}"
        )

    # Check content type if provided
    if file.content_type and not file.content_type.startswith('audio/'):
        raise HTTPException(
            400,
            f"Invalid content type: {file.content_type}. Expected audio/* content type"
        )


def validate_language(language: Optional[str]) -> None:
    """
    Validate language code.

    Raises HTTPException with 400 status for invalid language.
    """
    if not language:
        raise HTTPException(400, f"Language code is required. Supported languages: {', '.join(SUPPORTED_LANGUAGES)}")

    # Convert language code from "hi-IN" format to "hi" format if needed
    lang_code = language.split('-')[0] if '-' in language else language

    if lang_code not in SUPPORTED_LANGUAGES:
        raise HTTPException(
            400,
            f"Unsupported language: {lang_code}. Supported languages: {', '.join(SUPPORTED_LANGUAGES)}"
        )


app = FastAPI(
    title="ASR Service - IndicConformer",
    description="Multilingual ASR service for 22 Indian languages using AI4Bharat's IndicConformer model",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy" if asr and asr.model else "loading",
        "provider": "AI4Bharat IndicConformer",
        "model": MODEL_NAME,
        "device": asr.device if asr else "unknown",
        "decoder": DECODER_TYPE,
        "supported_languages": SUPPORTED_LANGUAGES
    }


@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe(
    file: UploadFile = File(...),
    language: str = Query(..., description="ISO 639-1 language code (e.g., 'hi', 'ta', 'hi-IN') - REQUIRED"),
    svc: ASRService = Depends(get_asr)
):
    """
    Transcribe audio file using IndicConformer model.

    Supported formats: WAV, FLAC, MP3, M4A, OGG, WebM
    Sample rate: Any (will be resampled to 16kHz)
    Channels: Mono or Stereo (will be converted to mono)
    Max file size: 100MB
    """
    # Validate inputs
    validate_audio_file(file)
    validate_language(language)

    try:
        # Read audio file
        audio_bytes = await file.read()

        # Validate file size
        if len(audio_bytes) > MAX_FILE_SIZE_BYTES:
            raise HTTPException(
                400,
                f"File too large: {len(audio_bytes) / 1024 / 1024:.2f}MB. Max size: {MAX_FILE_SIZE_MB}MB"
            )

        if len(audio_bytes) == 0:
            raise HTTPException(400, "Empty file uploaded")

        # Transcribe
        text, duration = svc.transcribe(audio_bytes, language_code=language)

        return TranscriptionResponse(
            text=text,
            language=svc.language,
            duration_seconds=round(duration, 2)
        )

    except HTTPException:
        # Re-raise HTTP exceptions (validation errors)
        raise
    except ValueError as e:
        # Audio processing errors (invalid format, corrupted file, etc.)
        raise HTTPException(400, str(e))
    except RuntimeError as e:
        # Transcription errors
        raise HTTPException(500, str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(500, "Internal server error")


@app.post("/transcribe/base64", response_model=TranscriptionResponse)
async def transcribe_b64(
    req: Base64AudioRequest,
    svc: ASRService = Depends(get_asr)
):
    """
    Transcribe base64 encoded audio using IndicConformer model.

    Request body:
    - audio_base64: Base64 encoded audio file
    - language_code: ISO 639-1 language code (e.g., 'hi', 'ta', 'hi-IN') - REQUIRED
    """
    # Validate language
    validate_language(req.language_code)

    try:
        # Decode base64 audio
        try:
            audio_bytes = base64.b64decode(req.audio_base64)
        except Exception as e:
            raise HTTPException(400, f"Invalid base64 encoding: {str(e)}")

        # Validate file size
        if len(audio_bytes) > MAX_FILE_SIZE_BYTES:
            raise HTTPException(
                400,
                f"File too large: {len(audio_bytes) / 1024 / 1024:.2f}MB. Max size: {MAX_FILE_SIZE_MB}MB"
            )

        if len(audio_bytes) == 0:
            raise HTTPException(400, "Empty audio data")

        # Transcribe
        text, duration = svc.transcribe(audio_bytes, language_code=req.language_code)

        return TranscriptionResponse(
            text=text,
            language=svc.language,
            duration_seconds=round(duration, 2)
        )

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except ValueError as e:
        # Audio processing errors
        raise HTTPException(400, str(e))
    except RuntimeError as e:
        # Transcription errors
        raise HTTPException(500, str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(500, "Internal server error")


@app.get("/languages")
async def list_languages():
    """List all supported languages."""
    return {
        "supported_languages": SUPPORTED_LANGUAGES,
        "decoder_type": DECODER_TYPE,
        "total_count": len(SUPPORTED_LANGUAGES)
    }


if __name__ == "__main__":
    import uvicorn
    print(f"\n{'='*60}")
    print(f"ASR Service - AI4Bharat IndicConformer")
    print(f"{'='*60}")
    print(f"API Documentation: http://{config.ASR_HOST}:{config.ASR_PORT}/docs")
    print(f"Health Check: http://{config.ASR_HOST}:{config.ASR_PORT}/health")
    print(f"Model: {MODEL_NAME}")
    print(f"Decoder: {DECODER_TYPE}")
    print(f"Supported Languages: {', '.join(SUPPORTED_LANGUAGES)}")
    print(f"{'='*60}\n")

    uvicorn.run(
        "task3_asr_service:app",
        host=config.ASR_HOST,
        port=config.ASR_PORT,
        reload=True
    )
