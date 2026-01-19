import argparse
import logging
import os
import re
import sys
from typing import List, Optional, Tuple

import chromadb
from sentence_transformers import SentenceTransformer

from config import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class WikipediaSectionChunker:
    """Chunks text by Wikipedia section headings.

    I used section based chunking instead of fixed size. Each chunk is a Wikipedia
    section with a target of around 4500 characters.

    Why section based: Wikipedia articles are already organized into meaningful
    sections, so preserving that structure keeps related information together.

    Why no overlap: Each section covers a distinct topic and is self contained,
    so overlap would just add redundancy without improving retrieval.
    """

    SECTION_PATTERN = re.compile(r'\n*={10,}\n(.+?)\n={10,}\n*')
    HEADER_FMT = "## {}\n\n"
    CONT_HEADER_FMT = "## {} (cont.)\n\n"

    def __init__(self, max_chars: int = None):
        self.max_chars = max_chars or config.MAX_CHUNK_CHARS

    def chunk(self, text: str) -> List[str]:
        if not text.strip():
            return []

        sections = self._split_into_sections(text)
        chunks = []
        for title, content in sections:
            chunks.extend(self._process_section(title, content))

        logger.info(f"Created {len(chunks)} chunks from {len(sections)} sections")
        return chunks

    def _split_into_sections(self, text: str) -> List[Tuple[str, str]]:
        parts = self.SECTION_PATTERN.split(text)
        sections = []

        if parts[0].strip():
            sections.append(("Introduction", parts[0].strip()))

        for i in range(1, len(parts), 2):
            title = parts[i].strip() if i < len(parts) else ""
            content = parts[i + 1].strip() if i + 1 < len(parts) else ""
            if title and content:
                sections.append((title, content))

        return sections or [("Content", text.strip())]

    def _process_section(self, title: str, content: str) -> List[str]:
        full_section = self.HEADER_FMT.format(title) + content
        if len(full_section) <= self.max_chars:
            return [full_section]
        return self._split_long_section(title, content)

    def _split_long_section(self, title: str, content: str) -> List[str]:
        chunks = []
        sentences = re.split(r'(?<=[.!?])\s+', content)
        header = self.HEADER_FMT.format(title)
        cont_header = self.CONT_HEADER_FMT.format(title)
        current = header

        for sentence in sentences:
            if len(current) + len(sentence) > self.max_chars:
                if current.strip():
                    chunks.append(current.strip())
                current = cont_header + sentence + " "
            else:
                current += sentence + " "

        if current.strip() and len(current) > len(cont_header):
            chunks.append(current.strip())

        return chunks


class SentenceTransformerEmbedder:
    """Generates embeddings using sentence-transformers."""

    def __init__(self, model_name: str = None):
        self.model_name = model_name or config.EMBEDDING_MODEL
        logger.info(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        self.dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Embedding dimension: {self.dim}")

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        return [e.tolist() for e in self.model.encode(texts, show_progress_bar=True)]

    def embed_single(self, text: str) -> List[float]:
        return self.model.encode(text).tolist()


class ChromaVectorStore:
    """ChromaDB vector store with persistence.

    I chose ChromaDB because it needs zero setup. No Docker, no cloud accounts,
    just pip install and it works. It also persists data to disk automatically.

    The tradeoffs are that it only runs on a single machine and has basic query
    features compared to Pinecone. But for a demo with one Wikipedia article at
    a time, that's fine.
    """

    def __init__(self, persist_dir: str = None, collection_name: str = None):
        self.persist_dir = persist_dir or config.CHROMA_PERSIST_DIR
        self.collection_name = collection_name or config.COLLECTION_NAME
        os.makedirs(self.persist_dir, exist_ok=True)

        self.client = chromadb.PersistentClient(path=self.persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info(f"ChromaDB ready at {self.persist_dir}, docs: {self.collection.count()}")

    def add(self, texts: List[str], embeddings: List[List[float]], metadata: List[dict]) -> None:
        if not texts:
            return
        self._clear_existing()
        ids = [f"doc_{i}" for i in range(len(texts))]
        self.collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadata)
        logger.info(f"Added {len(texts)} documents")

    def _clear_existing(self) -> None:
        if self.collection.count() > 0:
            existing = self.collection.get()["ids"]
            if existing:
                self.collection.delete(ids=existing)

    def query(self, embedding: List[float], n_results: int = 2) -> dict:
        return self.collection.query(
            query_embeddings=[embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )

    def count(self) -> int:
        return self.collection.count()


class VectorDatabaseBuilder:
    """Orchestrates chunking, embedding, and storage."""

    def __init__(self, chunker: WikipediaSectionChunker, embedder: SentenceTransformerEmbedder, store: ChromaVectorStore):
        self.chunker = chunker
        self.embedder = embedder
        self.store = store

    def build(self, text: str, source_name: str = "unknown") -> int:
        chunks = self.chunker.chunk(text)
        if not chunks:
            raise ValueError("No chunks created")

        embeddings = self.embedder.embed(chunks)
        metadata = [{"chunk_index": i, "chunk_size": len(c), "source": source_name} for i, c in enumerate(chunks)]
        self.store.add(chunks, embeddings, metadata)
        return len(chunks)


def load_file(path: str) -> Optional[str]:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    except IOError as e:
        logger.error(f"Read failed: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(prog='task2_vector_database')
    parser.add_argument('--input', '-i', type=str, default=os.path.join(config.OUTPUT_DIR, config.SCRAPED_TEXT_FILE))
    parser.add_argument('--max-chars', '-m', type=int, default=config.MAX_CHUNK_CHARS)
    parser.add_argument('--persist-dir', '-p', type=str, default=config.CHROMA_PERSIST_DIR)
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Input file not found: {args.input}")
        print("Run task1_data_collection.py first.")
        return 1

    text = load_file(args.input)
    if not text:
        return 1

    print(f"\n[1] Chunking by sections (max {args.max_chars} chars)")
    print(f"[2] Generating embeddings")
    print(f"[3] Storing in ChromaDB at {args.persist_dir}\n")

    builder = VectorDatabaseBuilder(
        WikipediaSectionChunker(args.max_chars),
        SentenceTransformerEmbedder(),
        ChromaVectorStore(args.persist_dir)
    )
    count = builder.build(text, os.path.basename(args.input))

    print(f"\nVector database created: {count} chunks at {args.persist_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
