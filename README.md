# Voice-Enabled RAG Chatbot

A multilingual voice chatbot that scrapes Wikipedia articles and answers questions about them. Supports voice input in Hindi, Tamil, Telugu, Bengali, Marathi, and Gujarati.

## What This Project Does

Basically, you give it a Wikipedia topic, it scrapes the article, and then you can ask questions about it - either by typing or speaking in one of the supported Indian languages. The app translates your voice to English, finds the relevant parts of the article, and generates an answer using Cohere's LLM.

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

Get your Sarvam API key from [sarvam.ai](https://sarvam.ai) (for translation) and Cohere key from [cohere.com](https://cohere.com) (for the LLM).

### 3. Run the App

You need two terminals:

**Terminal 1** - Start the ASR (speech recognition) service:
```bash
python task3_asr_service.py
```

**Terminal 2** - Start the main Gradio app:
```bash
python app.py
```

Then open the URL shown in Terminal 2 (usually `http://localhost:7860`).

## How to Use

1. Enter a Wikipedia topic and click "Scrape"
2. Wait for the article to be processed and stored
3. Either type your question or record voice input
4. Get your answer!

## Challenges & Implementation Decisions

### 1. Chunking Strategy
Choosing the right chunk size for Wikipedia articles was non-trivial. Very small chunks lose context, while large chunks reduce retrieval precision. The final approach uses ~4500 characters (≈500 tokens) per chunk, split by section boundaries. No overlap is used, as section-based chunking already preserves semantic coherence.

### 2. Scraping Robustness
Initial scraping and chunking would fail if the exact query term was not found on the page. A spell-check and query-normalization fallback was added to handle minor spelling variations and ambiguous inputs, improving overall robustness.

### 3. Rate Limiting
During testing, Cohere rate limits were hit. Basic retry logic with backoff was added to stabilize embedding generation.


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
| ASR | IndicWav2Vec | Open-source, optimized for Indian languages |
| Translation | Sarvam API | Built for Indian languages |

## Project Structure

```
├── app.py                    # Gradio UI 
├── config.py                 # Configuration settings
├── task1_data_collection.py  # Wikipedia scraping 
├── task2_vector_database.py  # Vector DB with ChromaDB 
├── task3_asr_service.py      # FastAPI ASR service
├── task4_translation.py      # Sarvam translation API 
├── task5_rag_pipeline.py     # End-to-end RAG pipeline 
├── requirements.txt          # Python dependencies
├── output/                   # Scraped Wikipedia articles
└── chroma_db/                # Vector database storage
```

## Notes

- This is a demo/prototype - not meant for production use
- ChromaDB persists data to disk in `chroma_db/` directory
- Voice input works best in a quiet environment

---

Built as part of a RAG implementation task. Feedback welcome!
