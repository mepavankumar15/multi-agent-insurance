"""
retriever.py — Query the in-memory VectorCollection for relevant policy clauses.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from .ingest import VectorCollection, _get_embedder


def retrieve_relevant_clauses(
    collection: VectorCollection,
    query_text: str,
    section_type: Optional[str] = None,
    k: int = 5,
) -> List[Dict]:
    """
    Embed the query and retrieve the top-k nearest clauses from the collection,
    optionally filtered by section_type metadata.

    Returns:
        List of {"text", "section_type", "source_page", "distance"}.
    """
    embedder = _get_embedder()
    query_emb = embedder.encode([query_text]).tolist()

    where = {"section_type": section_type} if section_type else None

    results = collection.query(
        query_embeddings=query_emb,
        n_results=k,
        where=where,
    )

    clauses: List[Dict] = []
    if results and results["documents"] and results["documents"][0]:
        docs = results["documents"][0]
        metas = results["metadatas"][0] if results["metadatas"] else [{}] * len(docs)
        dists = results["distances"][0] if results["distances"] else [0.0] * len(docs)

        for doc, meta, dist in zip(docs, metas, dists):
            clauses.append({
                "text": doc,
                "section_type": meta.get("section_type", "unknown"),
                "source_page": meta.get("source_page", -1),
                "distance": dist,
            })

    return clauses
