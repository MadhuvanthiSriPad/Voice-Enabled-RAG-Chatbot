"""Gradio UI for voice enabled RAG Chatbot."""

import os
import gradio as gr
import requests

from config import config
from task1_data_collection import collect, WikipediaAPISearch, WikipediaScraper
from task2_vector_database import (
    WikipediaSectionChunker, SentenceTransformerEmbedder,
    ChromaVectorStore, VectorDatabaseBuilder, load_file
)
from task5_rag_pipeline import process_audio_query, process_text_query


def scrape_and_build(topic: str) -> str:
    """Scrape Wikipedia and build vector database."""
    if not topic.strip():
        return "Please enter a topic"

    try:
        session = requests.Session()
        session.headers.update({"User-Agent": "VoiceRAGChatbot/1.0"})
        search_provider = WikipediaAPISearch(session)
        scraper = WikipediaScraper(session)
        output_path = os.path.join(config.OUTPUT_DIR, config.SCRAPED_TEXT_FILE)

        if not collect(topic, output_path, search_provider, scraper):
            return f"Failed to find/scrape Wikipedia article for: {topic}"

        text = load_file(output_path)
        if not text:
            return "Failed to read scraped content"

        builder = VectorDatabaseBuilder(
            WikipediaSectionChunker(),
            SentenceTransformerEmbedder(),
            ChromaVectorStore()
        )
        chunk_count = builder.build(text, topic)

        return f"Success: Scraped '{topic}' and created {chunk_count} chunks in vector database"
    except Exception as e:
        return f"Error: {e}"


def handle_text(text: str, history: list) -> tuple:
    """Process text input."""
    if not text.strip():
        return history, ""
    try:
        result = process_text_query(text)
        history.append((text, result["answer"]))
        context = "\n\n".join(f"[Chunk {i+1}]: {c}" for i, c in enumerate(result["chunks"]))
        return history, context
    except Exception as e:
        history.append((text, f"Error: {e}"))
        return history, ""


def handle_audio(audio_file, language: str, history: list) -> tuple:
    """Process audio input."""
    if audio_file is None:
        return history, "", ""
    try:
        result = process_audio_query(audio_file, language)
        question = f"[Audio] {result['transcribed']}"
        if result["translated"] != result["transcribed"]:
            question += f" -> {result['translated']}"
        history.append((question, result["answer"]))
        context = "\n\n".join(f"[Chunk {i+1}]: {c}" for i, c in enumerate(result["chunks"]))
        transcription = f"Transcribed: {result['transcribed']}\nTranslated: {result['translated']}"
        return history, transcription, context
    except Exception as e:
        history.append(("[Audio]", f"Error: {e}"))
        return history, str(e), ""


def clear():
    return [], "", "", "", ""


with gr.Blocks(title="Voice RAG Chatbot", theme=gr.themes.Soft()) as demo:
    gr.Markdown("## Voice-Enabled RAG Chatbot")
    gr.Markdown("*Ask questions about Wikipedia articles using text or voice in 22 Indian languages*")

    with gr.Row():
        with gr.Column(scale=2):
            with gr.Row():
                topic_input = gr.Textbox(label="Wikipedia Topic", placeholder="e.g., retrieval augmented generation", scale=3)
                scrape_btn = gr.Button("Build KB", variant="secondary", scale=1)
            scrape_status = gr.Textbox(label="Status", interactive=False, max_lines=1)

            chatbot = gr.Chatbot(label="Chat", height=250)

            with gr.Row():
                text_input = gr.Textbox(label="Question", placeholder="Type here...", scale=4)
                send_btn = gr.Button("Send", variant="primary", scale=1)

        with gr.Column(scale=1):
            audio_input = gr.Audio(label="Voice Input", type="filepath")
            language = gr.Dropdown(
                choices=[
                    ("Assamese", "as-IN"),
                    ("Bengali", "bn-IN"),
                    ("Bodo", "brx-IN"),
                    ("Dogri", "doi-IN"),
                    ("Gujarati", "gu-IN"),
                    ("Hindi", "hi-IN"),
                    ("Kannada", "kn-IN"),
                    ("Konkani", "kok-IN"),
                    ("Kashmiri", "ks-IN"),
                    ("Maithili", "mai-IN"),
                    ("Malayalam", "ml-IN"),
                    ("Manipuri", "mni-IN"),
                    ("Marathi", "mr-IN"),
                    ("Nepali", "ne-IN"),
                    ("Odia", "or-IN"),
                    ("Punjabi", "pa-IN"),
                    ("Sanskrit", "sa-IN"),
                    ("Santali", "sat-IN"),
                    ("Sindhi", "sd-IN"),
                    ("Tamil", "ta-IN"),
                    ("Telugu", "te-IN"),
                    ("Urdu", "ur-IN")
                ],
                value="hi-IN",
                label="Language"
            )
            audio_btn = gr.Button("Transcribe & Ask", variant="secondary")
            transcription = gr.Textbox(label="Transcription", lines=2)
            context_output = gr.Textbox(label="Retrieved Context", lines=3)
            clear_btn = gr.Button("Clear")

    # Events
    scrape_btn.click(scrape_and_build, [topic_input], [scrape_status])
    send_btn.click(handle_text, [text_input, chatbot], [chatbot, context_output]).then(lambda: "", outputs=[text_input])
    text_input.submit(handle_text, [text_input, chatbot], [chatbot, context_output]).then(lambda: "", outputs=[text_input])
    audio_btn.click(handle_audio, [audio_input, language, chatbot], [chatbot, transcription, context_output])
    clear_btn.click(clear, outputs=[chatbot, text_input, transcription, context_output, scrape_status])


if __name__ == "__main__":
    demo.launch()
