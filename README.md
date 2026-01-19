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

## Observations and Challenges

### Things That Worked Well

- **ChromaDB was a good choice** - I initially thought about using Pinecone or FAISS, but ChromaDB just works out of the box. No servers, no Docker, no hassle. For a demo project like this, it's perfect.

- **Sentence-transformers embeddings** - The `all-MiniLM-L6-v2` model is surprisingly good for its size. Retrieval quality was better than I expected.

### Challenges I Faced

1. **Voice input was tricky** - Getting Whisper to work reliably with Indian language accents took some trial and error. I had to add a fallback mechanism because sometimes the ASR service would timeout or return garbled text.

2. **Translation accuracy** - The Sarvam API works well for most inputs, but sometimes technical terms or code-mixed speech (Hindi + English) doesn't translate cleanly. This affects the final answer quality.

3. **Chunking strategy** - Figuring out the right chunk size for Wikipedia articles was harder than expected. Too small and you lose context, too large and retrieval becomes less precise. I settled on ~4500 characters (~500 tokens) chunked by section boundaries. No overlap is used since section-based chunking naturally preserves semantic coherence.

4. **Gradio audio component quirks** - The audio recording component behaves differently across browsers. Chrome works best, Firefox had some issues with the recording stopping early.

5. **Rate limits** - During testing, I hit Cohere's rate limits a few times. Had to add some basic retry logic.

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
├── app.py                    # Gradio UI (bonus task)
├── config.py                 # Configuration settings
├── task1_data_collection.py  # Wikipedia scraping (Task 1)
├── task2_vector_database.py  # Vector DB with ChromaDB (Task 2)
├── task3_asr_service.py      # FastAPI ASR service (Task 3)
├── task4_translation.py      # Sarvam translation API (Task 4)
├── task5_rag_pipeline.py     # End-to-end RAG pipeline (Task 5)
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