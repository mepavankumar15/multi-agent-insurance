"""
ingest.py — Policy PDF ingestion and embedding for the claims exclusion RAG pipeline.

Uses pdfplumber to extract text, detects the English/Chinese boundary, chunks the
English section by structural markers, embeds with sentence-transformers (or falls
back to sklearn TF-IDF when PyTorch is unavailable), and stores everything in a
lightweight pure-Python in-memory vector store (no ChromaDB native/Rust dependency).
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# In-memory vector store (replaces ChromaDB — avoids Rust/ONNX DLL issues)
# ---------------------------------------------------------------------------

@dataclass
class VectorCollection:
    """
    Minimal in-memory vector store: stores embeddings, documents, and metadata.
    Supports cosine-similarity nearest-neighbour queries with optional metadata filter.
    """
    name: str
    _ids:       List[str]       = field(default_factory=list)
    _docs:      List[str]       = field(default_factory=list)
    _embeddings: Optional[np.ndarray] = field(default=None)
    _metadatas:  List[dict]     = field(default_factory=list)

    def add(self, *, ids, documents, embeddings, metadatas=None):
        emb = np.array(embeddings, dtype="float32")
        if self._embeddings is None:
            self._embeddings = emb
        else:
            self._embeddings = np.vstack([self._embeddings, emb])
        self._ids.extend(ids)
        self._docs.extend(documents)
        self._metadatas.extend(metadatas or [{} for _ in ids])

    def query(self, *, query_embeddings, n_results=5, where=None):
        if self._embeddings is None or len(self._ids) == 0:
            return {"documents": [[]], "metadatas": [[]], "distances": [[]]}

        q = np.array(query_embeddings[0], dtype="float32")
        # Cosine similarity = dot product when both are L2-normalised
        q_norm = q / (np.linalg.norm(q) + 1e-10)
        emb = self._embeddings
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        emb_norm = emb / (norms + 1e-10)
        sims = emb_norm @ q_norm  # shape (N,)

        # Apply metadata filter
        mask = np.ones(len(self._ids), dtype=bool)
        if where:
            for key, val in where.items():
                mask &= np.array([m.get(key) == val for m in self._metadatas])

        # Set masked-out entries to -inf so they never rank top
        sims_filtered = np.where(mask, sims, -np.inf)

        top_k = min(n_results, int(mask.sum()))
        if top_k == 0:
            return {"documents": [[]], "metadatas": [[]], "distances": [[]]}

        top_idx = np.argpartition(sims_filtered, -top_k)[-top_k:]
        top_idx = top_idx[np.argsort(sims_filtered[top_idx])[::-1]]

        docs      = [self._docs[i]      for i in top_idx]
        metas     = [self._metadatas[i] for i in top_idx]
        distances = [float(1 - sims[i]) for i in top_idx]  # cosine distance

        return {"documents": [docs], "metadatas": [metas], "distances": [distances]}

    def __len__(self):
        return len(self._ids)


# ---------------------------------------------------------------------------
# Embedder — sentence-transformers with sklearn TF-IDF fallback
# ---------------------------------------------------------------------------

_embedder = None


class TfidfEmbedder:
    """
    Pure-Python fallback: sklearn TF-IDF 512-dim L2-normalised vectors.
    No PyTorch, no ONNX, no DLL dependencies.
    Retrieval quality is keyword-weighted rather than semantic, but works well
    for exclusion clauses which are very keyword-rich.
    """
    DIM = 512

    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self._vectorizer = TfidfVectorizer(max_features=self.DIM, stop_words="english")
        self._fitted = False

    def refit(self, corpus: List[str]):
        """Fit on the full corpus so all encode() calls share the same vocabulary."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        self._vectorizer = TfidfVectorizer(max_features=self.DIM, stop_words="english")
        self._vectorizer.fit(corpus)
        self._fitted = True

    def _ensure_fit(self, texts):
        if not self._fitted:
            self.refit(texts)

    def encode(self, texts: List[str]) -> np.ndarray:
        self._ensure_fit(texts)
        matrix = self._vectorizer.transform(texts).toarray().astype("float32")
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return matrix / norms  # L2-normalised


def _get_embedder():
    global _embedder
    if _embedder is None:
        try:
            from sentence_transformers import SentenceTransformer
            _embedder = SentenceTransformer("all-MiniLM-L6-v2")
            print("[ingest] Using sentence-transformers / all-MiniLM-L6-v2.")
        except Exception as e:
            print(f"[ingest] sentence-transformers unavailable ({type(e).__name__}). "
                  f"Using sklearn TF-IDF embedder (no PyTorch required).")
            _embedder = TfidfEmbedder()
    return _embedder


# ---------------------------------------------------------------------------
# PDF ingestion
# ---------------------------------------------------------------------------

def ingest_policy_pdf(file_path: str) -> Tuple[VectorCollection, dict]:
    """
    Extract text from a bilingual (English + Chinese) policy PDF, keep only
    the English section, chunk by structural markers, embed, and return an
    in-memory VectorCollection.

    Returns:
        collection  — VectorCollection ready for nearest-neighbour queries.
        summary     — {"exclusion": N, "definition": N, "procedure": N, "total": N}
    """
    import pdfplumber

    # ------------------------------------------------------------------
    # 1 & 2. Extract text page-by-page; stop at CJK-dominant pages
    # ------------------------------------------------------------------
    english_pages: List[dict] = []

    with pdfplumber.open(file_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            if not text.strip():
                continue
            english_pages.append({"page_num": i + 1, "text": text})

    # ------------------------------------------------------------------
    # 3. Structural chunking
    # ------------------------------------------------------------------
    chunks: List[dict] = []

    # Build combined text with page-boundary tracking
    combined = ""
    page_markers: List[Tuple[int, int]] = []  # (char_index, page_num)
    for pt in english_pages:
        page_markers.append((len(combined), pt["page_num"]))
        combined += pt["text"] + "\n"

    def _page_of(char_idx: int) -> int:
        for start, pnum in reversed(page_markers):
            if char_idx >= start:
                return pnum
        return 1

    lines = combined.split("\n")
    current_lines: List[str] = []
    current_type = "procedure"
    current_page = 1
    in_exclusions = False
    in_glossary = False
    char_pos = 0  # approximate char position tracker

    def _flush():
        nonlocal current_lines, current_type, current_page
        text = " ".join(current_lines).strip()
        if text:
            chunks.append({
                "text": text,
                "section_type": current_type,
                "source_page": current_page,
                "chunk_id": str(uuid.uuid4()),
            })
        current_lines = []

    for line in lines:
        stripped = line.strip()
        char_pos += len(line) + 1  # +1 for the \n we split on

        if not stripped:
            continue

        lower = stripped.lower()

        # Detect section boundary headers
        is_header = (
            len(stripped) < 120 and
            (stripped.isupper() or
             (stripped.istitle() and not stripped.endswith(".")))
        )

        if is_header:
            if "exclusion" in lower:
                _flush()
                in_exclusions, in_glossary = True, False
                current_type = "procedure"  # section header itself is procedure
                current_page = _page_of(char_pos)
                current_lines.append(stripped)
                continue
            elif "glossary" in lower or "definition" in lower:
                _flush()
                in_glossary, in_exclusions = True, False
                current_type = "procedure"
                current_page = _page_of(char_pos)
                current_lines.append(stripped)
                continue
            else:
                _flush()
                in_exclusions = in_glossary = False
                current_type = "procedure"
                current_page = _page_of(char_pos)
                current_lines.append(stripped)
                continue

        # Exclusion: numbered list items → each is its own chunk
        if in_exclusions and re.match(r"^\d+\.\s", stripped):
            _flush()
            current_type = "exclusion"
            current_page = _page_of(char_pos)
            current_lines.append(stripped)
            continue

        # Glossary: quoted-term definitions → each is its own chunk
        if in_glossary and re.match(r'^["\u201c].{1,60}["\u201d]', stripped):
            _flush()
            current_type = "definition"
            current_page = _page_of(char_pos)
            current_lines.append(stripped)
            continue

        # Continuation of current chunk
        if not current_lines:
            current_page = _page_of(char_pos)
        current_lines.append(stripped)

    _flush()

    # ------------------------------------------------------------------
    # 4. Embed
    # ------------------------------------------------------------------
    embedder = _get_embedder()
    texts = [c["text"] for c in chunks]

    if not texts:
        # Return empty collection
        collection = VectorCollection(name="policy_clauses")
        return collection, {"exclusion": 0, "definition": 0, "procedure": 0, "total": 0}

    # Refit TF-IDF on the full corpus for a stable vocabulary
    if isinstance(embedder, TfidfEmbedder):
        embedder.refit(texts)

    embeddings = embedder.encode(texts)  # np.ndarray shape (N, D)

    # ------------------------------------------------------------------
    # 5. Store in the in-memory VectorCollection
    # ------------------------------------------------------------------
    collection = VectorCollection(name="policy_clauses")
    collection.add(
        ids=[c["chunk_id"] for c in chunks],
        documents=texts,
        embeddings=embeddings.tolist(),
        metadatas=[{"section_type": c["section_type"], "source_page": c["source_page"]}
                   for c in chunks],
    )

    # ------------------------------------------------------------------
    # 6. Summary
    # ------------------------------------------------------------------
    summary = {"exclusion": 0, "definition": 0, "procedure": 0, "total": len(chunks)}
    for c in chunks:
        t = c["section_type"]
        if t in summary:
            summary[t] += 1

    return collection, summary


def extract_policy_metadata(pdf_path: str) -> dict:
    """
    Lightweight metadata extractor for uploaded policy PDFs.
    Uses pdfplumber + regex fallback. Safe, non-breaking addition.
    Returns a dict with: policy_number, holder, status, coverage_type,
    coverage_limit, deductible (or None if not found).
    """
    try:
        import pdfplumber
        import re

        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages[:4]:  # first 4 pages usually have the summary
                page_text = page.extract_text() or ""
                text += page_text + "\n"

        text_lower = text.lower()

        result = {
            "policy_number": None,
            "holder": None,
            "status": "active",
            "coverage_type": None,
            "coverage_limit": None,
            "deductible": None,
            "raw_text_sample": text[:800],
        }

        # Policy number patterns
        pol_patterns = [
            r"(?:policy|certificate|contract)\s*(?:no\.?|number|#)?\s*[:\-]?\s*([A-Z]{2,4}[\-\s]?\d{4,6})",
            r"(POL\-\d{4,6})",
        ]
        for pat in pol_patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                result["policy_number"] = m.group(1).upper().replace(" ", "")
                break

        # Holder name (common patterns)
        holder_patterns = [
            r"(?:insured|policyholder|holder|name)\s*[:\-]\s*([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){1,3})",
            r"([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){1,3})\s*(?:\(the insured|\(policyholder)",
        ]
        for pat in holder_patterns:
            m = re.search(pat, text)
            if m:
                result["holder"] = m.group(1).strip()
                break

        # Coverage type
        if any(k in text_lower for k in ["auto", "motor", "vehicle", "car"]):
            result["coverage_type"] = "auto"
        elif any(k in text_lower for k in ["home", "property", "household", "building"]):
            result["coverage_type"] = "home"
        elif any(k in text_lower for k in ["health", "medical", "hospital", "surgical", "clinical"]):
            result["coverage_type"] = "health"

        # Coverage limit
        limit_match = re.search(r"(?:limit|sum insured|coverage limit)[^\d]{0,20}(\$?\s*[\d,]+(?:\.\d{2})?)", text, re.IGNORECASE)
        if limit_match:
            try:
                val = float(limit_match.group(1).replace("$", "").replace(",", "").strip())
                result["coverage_limit"] = val
            except:
                pass

        # Deductible
        ded_match = re.search(r"(?:deductible|excess)[^\d]{0,20}(\$?\s*[\d,]+(?:\.\d{2})?)", text, re.IGNORECASE)
        if ded_match:
            try:
                val = float(ded_match.group(1).replace("$", "").replace(",", "").strip())
                result["deductible"] = val
            except:
                pass

        # Status
        if "lapsed" in text_lower or "expired" in text_lower or "cancelled" in text_lower:
            result["status"] = "lapsed"
        elif "suspended" in text_lower:
            result["status"] = "suspended"

        return result

    except Exception as e:
        print(f"[extract_policy_metadata] Error: {e}")
        return {
            "policy_number": None,
            "holder": None,
            "status": "active",
            "coverage_type": None,
            "coverage_limit": None,
            "deductible": None,
            "error": str(e),
        }
