# Voice-Enabled RAG Chatbot

A multilingual voice chatbot that scrapes Wikipedia articles and answers questions about them. Supports voice input in **Hindi and Tamil**.

## What This Project Does

Basically, you give it a Wikipedia topic, it scrapes the article, and then you can ask questions about it - either by typing or speaking in Hindi or Tamil. The app uses Sarvam AI to transcribe your voice, translates it to English, finds the relevant parts of the article, and generates an answer using Cohere's LLM.

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up Your API Keys

You'll need two API keys:

```bash
export SARVAM_API_KEY="your_sarvam_key"
export LLM_API_KEY="your_cohere_key"
```

Get your Sarvam API key from [sarvam.ai](https://sarvam.ai) (for speech-to-text and translation) and Cohere key from [cohere.com](https://cohere.com) (for the LLM).

### 3. Run the App

Navigate to the `voice_rag` directory and run two terminals:

**Terminal 1** - Start the ASR (speech recognition) service:
```bash
cd voice_rag
python task3_asr_service.py
```

**Terminal 2** - Start the main Gradio app:
```bash
cd voice_rag
python app.py
```

Then open the URL shown in Terminal 2 (usually `http://localhost:7860`).

## How to Use

1. Enter a Wikipedia topic (e.g., "Franz Kafka") and click "Build KB"
2. Wait for the article to be scraped and indexed
3. Select your language (Hindi or Tamil) from the dropdown
4. Either type your question in English or record voice input in Hindi/Tamil
5. Get your answer with the retrieved context!

## Challenges & Implementation Decisions

### 1. Chunking Strategy
Choosing the right chunk size for Wikipedia articles was non-trivial. Very small chunks lose context, while large chunks reduce retrieval precision. The final approach uses ~4500 characters (≈500 tokens) per chunk, split by section boundaries. No overlap is used, as section-based chunking already preserves semantic coherence.

### 2. Scraping Robustness
Initial scraping and chunking would fail if the exact query term was not found on the page. A spell-check and query-normalization fallback was added to handle minor spelling variations and ambiguous inputs, improving overall robustness.



### What I'd Do Differently

- Add caching for repeated queries
- Support multiple Wikipedia articles at once
- Add a "confidence score" to show how relevant the retrieved chunks are
- Maybe try a local LLM option for offline use

## Tech Stack

| Component | What I Used | Why |
|-----------|-------------|-----|
| Frontend | Gradio | Quick to set up, handles audio well |
| Vector DB | ChromaDB | Simple, no external dependencies |
| Embeddings | sentence-transformers | Good quality, runs locally |
| LLM | Cohere | Generous free tier, good results |
| ASR | Sarvam AI | Excellent for Indian languages (Hindi, Tamil) |
| Translation | Sarvam API | Built for Indian languages |

## Project Structure

```
voice_rag/
├── app.py                    # Gradio UI
├── config.py                 # Configuration settings
├── task1_data_collection.py  # Wikipedia scraping
├── task2_vector_database.py  # Vector DB with ChromaDB
├── task3_asr_service.py      # FastAPI ASR service (Sarvam AI)
├── task4_translation.py      # Sarvam translation API
├── task5_rag_pipeline.py     # End-to-end RAG pipeline
├── output/                   # Scraped Wikipedia articles
└── chroma_db/                # Vector database storage

requirements.txt              # Python dependencies
README.md                     # This file
```

## Pipeline

The complete voice RAG pipeline works as follows:

1. **Audio Input** → User speaks in Hindi or Tamil
2. **ASR (Sarvam AI)** → Transcribes audio to text in the original language
3. **Translation (Sarvam AI)** → Translates to English
4. **Retrieval** → Finds relevant chunks from the Wikipedia article using semantic search
5. **LLM (Cohere)** → Generates an answer based on the retrieved context

---


Feedback welcome!
