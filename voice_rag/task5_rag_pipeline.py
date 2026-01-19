"""Task 5: End-to-end Voice RAG Pipeline
Audio -> ASR -> Translation -> Retrieval -> LLM -> Answer
"""

import requests
from config import config

# Import from previous tasks
from task2_vector_database import SentenceTransformerEmbedder, ChromaVectorStore
from task4_translation import translate_to_english

# Initialize from task2
embedder = SentenceTransformerEmbedder()
vector_store = ChromaVectorStore()


def transcribe_audio(audio_path: str) -> str:
    """Step 1: Call ASR service (task3) endpoint."""
    with open(audio_path, 'rb') as f:
        response = requests.post(
            f"http://{config.ASR_HOST}:{config.ASR_PORT}/transcribe",
            files={'file': f},
            timeout=60
        )
    response.raise_for_status()
    return response.json().get('text', '')


def retrieve_chunks(query: str, top_k: int = 2) -> list:
    """Step 3: Retrieve chunks using task2 components."""
    query_embedding = embedder.embed_single(query)
    results = vector_store.query(query_embedding, n_results=top_k)
    return results.get('documents', [[]])[0]


def generate_answer(query: str, chunks: list) -> str:
    """Step 4: Generate answer using Cohere LLM API."""
    context = "\n\n".join(chunks)
    prompt = f"""Based on the following context, answer the question.
If the answer is not in the context, say so.

Context:
{context}

Question: {query}

Answer:"""

    response = requests.post(
        "https://api.cohere.com/v1/chat",
        headers={
            "Authorization": f"Bearer {config.LLM_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "command-a-03-2025",
            "message": prompt
        },
        timeout=60
    )
    response.raise_for_status()
    return response.json().get("text", "")


def process_audio_query(audio_path: str, source_language: str = "hi-IN") -> dict:
    """Full pipeline: Audio -> ASR (task3) -> Translation (task4) -> Retrieval (task2) -> LLM -> Answer"""
    transcribed = transcribe_audio(audio_path)
    print(f"Transcribed: {transcribed}")

    translated = translate_to_english(transcribed, source_language)
    print(f"Translated: {translated}")

    chunks = retrieve_chunks(translated, top_k=2)
    print(f"Retrieved {len(chunks)} chunks")

    answer = generate_answer(translated, chunks)
    return {"transcribed": transcribed, "translated": translated, "chunks": chunks, "answer": answer}


def process_text_query(text: str) -> dict:
    """Process text query (skip ASR and translation)."""
    chunks = retrieve_chunks(text, top_k=2)
    answer = generate_answer(text, chunks)
    return {"query": text, "chunks": chunks, "answer": answer}


if __name__ == "__main__":
    # Example: test with a text query
    result = process_text_query("What is retrieval augmented generation?")
    print(f"\nAnswer: {result['answer']}")
