"""Configuration for Voice-Enabled RAG Chatbot."""

import os
from dataclasses import dataclass


@dataclass
class Config:
    # API Keys
    SARVAM_API_KEY: str = os.getenv("SARVAM_API_KEY", "")
    LLM_API_KEY: str = os.getenv("LLM_API_KEY", "")

    # Vector Database
    CHROMA_PERSIST_DIR: str = "./chroma_db"
    COLLECTION_NAME: str = "wikipedia_articles"
    MAX_CHUNK_CHARS: int = 4500

    # Models
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

    # ASR Service
    ASR_HOST: str = "localhost"
    ASR_PORT: int = 8000

    # Supported Languages (22 Indian languages)
    SUPPORTED_LANGUAGES: list = (
        "as-IN", "bn-IN", "brx-IN", "doi-IN", "gu-IN", "hi-IN", "kn-IN", "kok-IN",
        "ks-IN", "mai-IN", "ml-IN", "mni-IN", "mr-IN", "ne-IN", "or-IN", "pa-IN",
        "sa-IN", "sat-IN", "sd-IN", "ta-IN", "te-IN", "ur-IN"
    )

    # Output
    OUTPUT_DIR: str = "./output"
    SCRAPED_TEXT_FILE: str = "wikipedia_article.txt"


config = Config()
