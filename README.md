# Voice-Enabled RAG Chatbot

A multilingual voice chatbot that scrapes Wikipedia articles and answers questions about them. Supports voice input in **22 Indian languages**: Assamese, Bengali, Bodo, Dogri, Gujarati, Hindi, Kannada, Konkani, Kashmiri, Maithili, Malayalam, Manipuri, Marathi, Nepali, Odia, Punjabi, Sanskrit, Santali, Sindhi, Tamil, Telugu, and Urdu.

## What This Project Does

Basically, you give it a Wikipedia topic, it scrapes the article, and then you can ask questions about it - either by typing or speaking in any of the 22 supported Indian languages. The app uses AI4Bharat's IndicConformer model (hosted locally) to transcribe your voice, translates it to English using Sarvam AI, finds the relevant parts of the article, and generates an answer using Cohere's LLM.

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r voice_rag/requirements.txt
```

### 2. Request Access to IndicConformer Model

1. Go to [https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual](https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual)
2. Click **"Request access"** or **"Accept conditions"**
3. Create a Hugging Face token at [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
4. Login via CLI:
   ```bash
   pip install huggingface-hub
   huggingface-cli login
   ```

### 3. Set Up Your API Keys

You'll need two API keys:

```bash
export SARVAM_API_KEY="your_sarvam_key"  # For translation only
export LLM_API_KEY="your_cohere_key"
```

Get your Sarvam API key from [sarvam.ai](https://sarvam.ai) (for translation) and Cohere key from [cohere.com](https://cohere.com) (for the LLM). Note: ASR now runs locally using IndicConformer, so no API key is needed for speech recognition.

### 4. Run the App

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
4. Either type your question in English or record voice input in any of the 22 supported Indian languages
5. Get your answer with the retrieved context!

**Supported Languages for Voice Input (22 languages):**
1. Assamese (as)
2. Bengali (bn)
3. Bodo (brx)
4. Dogri (doi)
5. Gujarati (gu)
6. Hindi (hi)
7. Kannada (kn)
8. Konkani (kok)
9. Kashmiri (ks)
10. Maithili (mai)
11. Malayalam (ml)
12. Manipuri (mni)
13. Marathi (mr)
14. Nepali (ne)
15. Odia (or)
16. Punjabi (pa)
17. Sanskrit (sa)
18. Santali (sat)
19. Sindhi (sd)
20. Tamil (ta)
21. Telugu (te)
22. Urdu (ur)

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
| ASR | AI4Bharat IndicConformer | State-of-the-art model for 22 Indian languages, runs locally |
| Translation | Sarvam API | Built for Indian languages |

## Project Structure

```
voice_rag/
├── app.py                    # Gradio UI
├── config.py                 # Configuration settings
├── task1_data_collection.py  # Wikipedia scraping
├── task2_vector_database.py  # Vector DB with ChromaDB
├── task3_asr_service.py      # FastAPI ASR service (IndicConformer)
├── task4_translation.py      # Sarvam translation API
├── task5_rag_pipeline.py     # End-to-end RAG pipeline
├── output/                   # Scraped Wikipedia articles
└── chroma_db/                # Vector database storage

requirements.txt              # Python dependencies
README.md                     # This file
```

## Pipeline

The complete voice RAG pipeline works as follows:

1. **Audio Input** → User speaks in any of 22 supported Indian languages
2. **ASR (IndicConformer)** → Transcribes audio to text in the original language using locally-hosted model
3. **Translation (Sarvam AI)** → Translates to English
4. **Retrieval** → Finds relevant chunks from the Wikipedia article using semantic search
5. **LLM (Cohere)** → Generates an answer based on the retrieved context

---


Feedback welcome!
