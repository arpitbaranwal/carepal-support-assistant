"""
Text Chunker, Vector Store Ingestion, and Retrieval Pipeline (Phase 1).
Reads clean scraped pages from data/raw_pages.json, applies semantic recursive
chunking, attaches rich page-level citation metadata, computes dense embeddings
using SentenceTransformers, and persists a local FAISS vector index.
"""

import sys
import os
import json
import time
import re
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config

# Set stdout encoding to UTF-8 on Windows
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


class RecursiveTextChunker:
    """
    Splits long documents along natural paragraph, sentence, and word boundaries
    while enforcing chunk size limits and preserving overlap.
    """

    def __init__(
        self,
        chunk_size: int = config.CHUNK_SIZE,
        chunk_overlap: int = config.CHUNK_OVERLAP,
        separators: Optional[List[str]] = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or config.CHUNK_SEPARATORS

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """Recursively split text using the first matching separator."""
        final_chunks: List[str] = []
        separator = separators[-1]
        new_separators = []

        for i, _s in enumerate(separators):
            if _s == "":
                separator = _s
                break
            if _s in text:
                separator = _s
                new_separators = separators[i + 1:]
                break

        splits = text.split(separator) if separator else list(text)

        good_splits: List[str] = []
        for s in splits:
            if not s.strip():
                continue
            if len(s) < self.chunk_size:
                good_splits.append(s)
            else:
                if new_separators:
                    sub_splits = self._split_text(s, new_separators)
                    good_splits.extend(sub_splits)
                else:
                    good_splits.append(s)

        # Merge splits respecting chunk_size and chunk_overlap
        current_chunk: List[str] = []
        current_len = 0

        for s in good_splits:
            s_len = len(s) + (len(separator) if current_chunk else 0)
            if current_len + s_len <= self.chunk_size:
                current_chunk.append(s)
                current_len += s_len
            else:
                if current_chunk:
                    joined = separator.join(current_chunk).strip()
                    if joined:
                        final_chunks.append(joined)
                    # Handle overlap by keeping tail elements
                    overlap_len = 0
                    overlap_chunk = []
                    for piece in reversed(current_chunk):
                        if overlap_len + len(piece) <= self.chunk_overlap:
                            overlap_chunk.insert(0, piece)
                            overlap_len += len(piece)
                        else:
                            break
                    current_chunk = overlap_chunk
                    current_len = sum(len(p) for p in current_chunk) + (len(separator) * max(0, len(current_chunk) - 1))

                current_chunk.append(s)
                current_len += len(s)

        if current_chunk:
            joined = separator.join(current_chunk).strip()
            if joined:
                final_chunks.append(joined)

        return final_chunks

    def chunk_page(self, page_dict: Dict) -> List[Dict]:
        """
        Splits a page into structured chunks enriched with citation metadata.
        """
        url = page_dict.get("url", "")
        title = page_dict.get("title", "The CarePal")
        category = page_dict.get("category", "general")
        label = page_dict.get("label", "")
        content = page_dict.get("content", "")

        slug = url.rstrip("/").split("/")[-1] or "home"
        slug_clean = re.sub(r"[^a-zA-Z0-9_-]", "_", slug)

        raw_chunks = self._split_text(content, self.separators)
        total_chunks = len(raw_chunks)

        processed = []
        for idx, chunk_text in enumerate(raw_chunks):
            chunk_id = f"{slug_clean}_chunk_{idx + 1:03d}"
            processed.append({
                "chunk_id": chunk_id,
                "page_url": url,
                "page_title": title,
                "category": category,
                "label": label,
                "chunk_index": idx,
                "total_chunks_in_page": total_chunks,
                "char_count": len(chunk_text),
                "word_count": len(chunk_text.split()),
                "content": chunk_text,
            })
        return processed


class CarePalVectorStore:
    """Manages local FAISS vector indexing, persistence, and semantic retrieval."""

    def __init__(
        self,
        model_name: str = config.EMBEDDING_MODEL_NAME,
        device: str = config.EMBEDDING_DEVICE,
    ):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.index = None
        self.metadata: List[Dict] = []

    def _load_model(self):
        """Lazy load SentenceTransformers model (fast local cache first)."""
        if self.model is None:
            print(f"Loading embedding model '{self.model_name}' on {self.device}...")
            from sentence_transformers import SentenceTransformer
            try:
                # Fast load from local cache without online HuggingFace ping/retries
                self.model = SentenceTransformer(self.model_name, device=self.device, local_files_only=True)
            except Exception:
                self.model = SentenceTransformer(self.model_name, device=self.device)
            print("Embedding model loaded successfully.")

    def build_and_save(
        self,
        chunks: List[Dict],
        index_path: Path = config.VECTOR_INDEX_FILE,
        metadata_path: Path = config.METADATA_FILE,
        chunks_path: Path = config.CHUNKS_FILE,
    ) -> None:
        """
        Computes dense embeddings for chunks, constructs FAISS IndexFlatIP
        with normalized vectors (cosine similarity), and saves to disk.
        """
        import faiss

        self._load_model()
        texts = [c["content"] for c in chunks]

        print(f"Generating embeddings for {len(texts)} chunks...")
        start_time = time.time()
        # Compute embeddings with sentence-transformers
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=config.NORMALIZE_EMBEDDINGS,
        )
        embeddings = np.array(embeddings, dtype=np.float32)
        embed_time = time.time() - start_time
        print(f"Computed {embeddings.shape[0]} embeddings (dim={embeddings.shape[1]}) in {embed_time:.2f}s.")

        dimension = embeddings.shape[1]

        # Inner Product on L2-normalized vectors == Cosine Similarity
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)
        print(f"FAISS index built. Total indexed vectors: {index.ntotal}")

        # Persist FAISS index binary
        index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(index_path))
        print(f"Saved FAISS index to: {index_path}")

        # Persist metadata JSON
        metadata_list = []
        for i, c in enumerate(chunks):
            metadata_list.append({
                "index_id": i,
                "chunk_id": c["chunk_id"],
                "page_url": c["page_url"],
                "page_title": c["page_title"],
                "category": c["category"],
                "label": c["label"],
                "chunk_index": c["chunk_index"],
                "total_chunks_in_page": c["total_chunks_in_page"],
                "char_count": c["char_count"],
                "word_count": c["word_count"],
            })

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata_list, f, indent=2, ensure_ascii=False)
        print(f"Saved metadata to: {metadata_path}")

        # Persist full chunks JSON
        with open(chunks_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)
        print(f"Saved chunks to: {chunks_path}")

        self.index = index
        self.metadata = chunks

    def load(
        self,
        index_path: Path = config.VECTOR_INDEX_FILE,
        chunks_path: Path = config.CHUNKS_FILE,
    ) -> bool:
        """Loads persistent FAISS index and chunk metadata from disk."""
        import faiss

        if not index_path.exists() or not chunks_path.exists():
            print(f"[Error] Index file or chunks file missing from {config.VECTOR_DB_DIR}")
            return False

        print(f"Loading FAISS index from {index_path}...")
        self.index = faiss.read_index(str(index_path))

        print(f"Loading chunk metadata from {chunks_path}...")
        with open(chunks_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        print(f"Vector store loaded successfully: {self.index.ntotal} vectors indexed.")
        return True

    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Executes semantic search over the FAISS vector store.
        Returns top_k chunks with cosine similarity score and citation metadata.
        """
        if self.index is None or not self.metadata:
            if not self.load():
                return []

        self._load_model()

        # Embed query and normalize
        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=config.NORMALIZE_EMBEDDINGS,
        )
        query_vec = np.array(query_embedding, dtype=np.float32)

        # FAISS search
        scores, indices = self.index.search(query_vec, top_k)

        results = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
            if idx < 0 or idx >= len(self.metadata):
                continue
            chunk_data = self.metadata[idx]
            results.append({
                "rank": rank,
                "score": float(score),
                "chunk_id": chunk_data.get("chunk_id"),
                "page_title": chunk_data.get("page_title"),
                "page_url": chunk_data.get("page_url"),
                "category": chunk_data.get("category"),
                "label": chunk_data.get("label"),
                "content": chunk_data.get("content"),
            })
        return results


def run_pipeline() -> None:
    """Executes end-to-end chunking, embedding, vector ingestion, and verification."""
    print("=" * 80)
    print("CarePal Vector Store Ingestion Pipeline (Phase 1)")
    print("=" * 80)

    # 1. Load raw pages
    if not config.RAW_PAGES_FILE.exists():
        print(f"[Error] Raw pages file not found at: {config.RAW_PAGES_FILE}")
        print("Please run `python src/scraper.py` first.")
        sys.exit(1)

    with open(config.RAW_PAGES_FILE, "r", encoding="utf-8") as f:
        raw_pages = json.load(f)

    print(f"Loaded {len(raw_pages)} scraped pages from {config.RAW_PAGES_FILE}")

    # 2. Chunking
    chunker = RecursiveTextChunker(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )

    all_chunks: List[Dict] = []
    for page in raw_pages:
        page_chunks = chunker.chunk_page(page)
        all_chunks.extend(page_chunks)

    print(f"Generated {len(all_chunks)} semantic chunks (chunk_size={config.CHUNK_SIZE}, overlap={config.CHUNK_OVERLAP}).")

    # Save processed chunks
    with open(config.PROCESSED_CHUNKS_FILE, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)
    print(f"Saved processed chunks for inspection at: {config.PROCESSED_CHUNKS_FILE}")

    # 3. Vector Ingestion
    vector_store = CarePalVectorStore(
        model_name=config.EMBEDDING_MODEL_NAME,
        device=config.EMBEDDING_DEVICE,
    )
    vector_store.build_and_save(all_chunks)

    # 4. End-to-End Sanity & Citation Verification Suite
    print("\n" + "=" * 80)
    print("Executing Vector Store Retrieval & Citation Verification Suite")
    print("=" * 80)

    test_queries = [
        (
            "What is the refund policy for CarePal services?",
            "refund",
            "Refund Policy",
        ),
        (
            "What does The CarePal elderly care service include?",
            "elderly",
            "Elderly Care",
        ),
        (
            "Do you provide 24/7 full-time live-in caregivers and dementia memory care?",
            "livein",
            "Live-in Care / Dementia",
        ),
        (
            "How can I consult a doctor or get emergency medical assistance?",
            "doctor",
            "Doctor Consultation / Emergency",
        ),
    ]

    for q_idx, (query, expected_keyword, intent) in enumerate(test_queries, start=1):
        print(f"\n[Test Query {q_idx}] Intent: {intent}")
        print(f"Question: \"{query}\"")
        results = vector_store.search(query, top_k=2)

        if not results:
            print("  ✗ No results returned!")
            continue

        top_match = results[0]
        url = top_match["page_url"]
        title = top_match["page_title"]
        score = top_match["score"]
        snippet = top_match["content"][:200].replace("\n", " ")

        passed = expected_keyword in url.lower() or expected_keyword in title.lower()
        status_badge = "[PASS]" if passed else "[WARN]"

        print(f"  {status_badge} Top Match (Cosine Sim: {score:.4f})")
        print(f"    • Cited Page Title : {title}")
        print(f"    • Cited URL        : {url}")
        print(f"    • Chunk Content    : {snippet}...")

    print("\n" + "=" * 80)
    print("Ingestion Pipeline Completed Successfully!")
    print(f"• Raw Pages       : {len(raw_pages)} pages in {config.RAW_PAGES_FILE}")
    print(f"• Processed Chunks: {len(all_chunks)} chunks in {config.PROCESSED_CHUNKS_FILE}")
    print(f"• Persistent Index: {config.VECTOR_INDEX_FILE}")
    print(f"• Chunks Metadata : {config.CHUNKS_FILE}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Ingest CarePal data into local FAISS vector store.")
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Execute a test search query against the vector database",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of retrieved search results",
    )
    args = parser.parse_args()

    if args.query:
        store = CarePalVectorStore()
        results = store.search(args.query, top_k=args.top_k)
        print(f"\nSearch results for query: \"{args.query}\"")
        print("=" * 80)
        for r in results:
            print(f"Rank {r['rank']} | Score: {r['score']:.4f} | Category: {r['category']}")
            print(f"Title: {r['page_title']}")
            print(f"URL: {r['page_url']}")
            print(f"Content snippet:\n{r['content']}")
            print("-" * 80)
    else:
        run_pipeline()


if __name__ == "__main__":
    main()
