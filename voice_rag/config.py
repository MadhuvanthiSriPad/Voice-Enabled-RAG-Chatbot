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
    ASR_MODEL_NAME: str = "ai4bharat/indicwav2vec_v1_hindi"

    # ASR Service
    ASR_HOST: str = "localhost"
    ASR_PORT: int = 8000

    # Output
    OUTPUT_DIR: str = "./output"
    SCRAPED_TEXT_FILE: str = "wikipedia_article.txt"


config = Config()
