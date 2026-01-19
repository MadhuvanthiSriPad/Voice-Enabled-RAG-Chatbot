"""Task 4: Translate text to English using Sarvam API."""

import os
import requests

API_URL = "https://api.sarvam.ai/translate"
API_KEY = os.getenv("SARVAM_API_KEY", "")

SUPPORTED_LANGUAGES = [
    "hi-IN",  # Hindi
    "ta-IN"   # Tamil
]


def translate_to_english(text: str, source_language: str = "hi-IN") -> str:
    """Translate text to English using Sarvam API."""
    if not API_KEY:
        raise ValueError("SARVAM_API_KEY environment variable not set")

    response = requests.post(
        API_URL,
        json={
            "input": text,
            "source_language_code": source_language,
            "target_language_code": "en-IN",
            "speaker_gender": "Male",
            "mode": "formal",
            "enable_preprocessing": True
        },
        headers={
            "Content-Type": "application/json",
            "api-subscription-key": API_KEY
        },
        timeout=30
    )
    response.raise_for_status()
    return response.json().get("translated_text", "")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python task4_translation.py <text> [source_language]")
        print(f"Supported languages: {', '.join(SUPPORTED_LANGUAGES)}")
        sys.exit(1)

    text = sys.argv[1]
    source = sys.argv[2] if len(sys.argv) > 2 else "hi-IN"

    result = translate_to_english(text, source)
    print(f"Original: {text}")
    print(f"Translated: {result}")
