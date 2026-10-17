"""
search.py — The Naive RAG Evidence Retrieval Stage (Stage 5+6).

The Goal:
    Given a natural-language question, generate its embedding using the
    configured NVIDIA NIM model, query the Pinecone index with that vector,
    and return the top matching chunks — including original text, file path,
    chunk index, and cosine similarity score — as structured evidence.

The "Why":
    This is the core retrieval function that bridges a user's question to
    the stored knowledge base. It retrieves *evidence only* — no AI-generated
    answer is produced here. Downstream stages (context injection + generation)
    consume these results to produce grounded answers.

Design Decisions:
    - Uses the existing embedding module (embed_text) and vector_store module
      (query_vector_db) — no LangChain, no LlamaIndex.
    - Validates all returned data before surfacing it to the caller.
    - Returns an empty list (never crashes) when no results match.
    - Assigns a relevance tier label (HIGH / MODERATE / LOW / NOISE) to each
      result so callers can filter programmatically.
    - Never prints or logs API keys.
"""

from typing import Any, Dict, List, Optional

from config import RAGConfig, get_rag_config
from embedding import embed_text, EXPECTED_DIMENSION, EmbeddingError
from vector_store import (
    query_vector_db,
    VectorStoreError,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_TOP_K: int = 5

# Cosine-similarity score thresholds (nvidia/nemotron-3-embed-1b range)
SCORE_HIGH: float = 0.45
SCORE_MODERATE: float = 0.25
SCORE_LOW: float = 0.10


# ---------------------------------------------------------------------------
# Custom Exceptions
# ---------------------------------------------------------------------------
class SearchError(Exception):
    """Base exception for retrieval-stage failures."""
    pass


class SearchInputError(SearchError):
    """Raised when the search query is invalid (empty, wrong type)."""
    pass


# ---------------------------------------------------------------------------
# Result Validation & Enrichment
# ---------------------------------------------------------------------------
def _relevance_tier(score: float) -> str:
    """
    Assign a human-readable relevance tier based on cosine similarity.

    Thresholds were calibrated against nvidia/nemotron-3-embed-1b where
    a score of ~0.5+ indicates strong topical overlap, 0.25-0.45 moderate
    relatedness, and below 0.10 is essentially noise.
    """
    if score >= SCORE_HIGH:
        return "HIGH"
    if score >= SCORE_MODERATE:
        return "MODERATE"
    if score >= SCORE_LOW:
        return "LOW"
    return "NOISE"


def _validate_and_enrich(raw_match: Dict[str, Any], rank: int) -> Dict[str, Any]:
    """
    Validate a single raw match from the vector store and produce a
    normalized evidence record with all expected fields.

    Missing fields are replaced with safe defaults rather than crashing.
    """
    match_id = raw_match.get("id", f"unknown_{rank}")
    text = raw_match.get("text", "")
    score = raw_match.get("score")
    metadata = raw_match.get("metadata", {})

    # Parse source_file and chunk_index from metadata or from the ID
    source_file = metadata.get("source_file", "")
    chunk_index_str = metadata.get("chunk_index")
    chunk_index: Optional[int] = None

    if chunk_index_str is not None:
        try:
            chunk_index = int(chunk_index_str)
        except (ValueError, TypeError):
            chunk_index = None

    # Fallback: parse file_path and chunk_index from IDs like "config.py#chunk_3"
    if not source_file and "#chunk_" in match_id:
        parts = match_id.rsplit("#chunk_", 1)
        source_file = parts[0]
        if chunk_index is None:
            try:
                chunk_index = int(parts[1])
            except (ValueError, IndexError):
                pass

    # Fallback for doc_chunk_N style IDs
    if not source_file and match_id.startswith("doc_chunk_"):
        source_file = "document"
        if chunk_index is None:
            try:
                chunk_index = int(match_id.replace("doc_chunk_", ""))
            except ValueError:
                pass

    # Validate score
    if score is None or not isinstance(score, (int, float)):
        score = 0.0

    # Validate text is a non-empty string
    if not isinstance(text, str):
        text = str(text) if text is not None else ""

    return {
        "rank": rank,
        "id": match_id,
        "text": text,
        "source_file": source_file,
        "chunk_index": chunk_index,
        "score": float(score),
        "relevance": _relevance_tier(float(score)),
    }


# ---------------------------------------------------------------------------
# Core Search Function
# ---------------------------------------------------------------------------
def search(
    question: str,
    config: Optional[RAGConfig] = None,
    top_k: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Stage 5+6: Embed a question and retrieve the most relevant evidence chunks.

    This function performs evidence retrieval only — it does NOT generate an
    AI answer. Downstream pipeline stages consume these results for context
    injection and grounded generation.

    Args:
        question: The natural-language question to search for.
        config: RAG configuration. Loaded from environment if None.
        top_k: Number of top results to retrieve. Uses config.top_k if None.

    Returns:
        List of evidence dicts, each containing:
            - rank (int): 1-based position in the result set
            - id (str): Unique chunk identifier
            - text (str): The original chunk text
            - source_file (str): File path where the chunk originated
            - chunk_index (int | None): Position within the source file
            - score (float): Cosine similarity score (0.0 – 1.0)
            - relevance (str): Tier label — HIGH / MODERATE / LOW / NOISE

        Returns an empty list if no matches are found or the index is empty.

    Raises:
        SearchInputError: If the question is empty, whitespace-only, or wrong type.
        SearchError: If embedding or vector store calls fail unexpectedly.
    """
    # ── 1. Validate input ────────────────────────────────────────────────
    if not isinstance(question, str):
        raise SearchInputError(
            f"Question must be a string, got {type(question).__name__}."
        )

    cleaned = question.strip()
    if not cleaned:
        raise SearchInputError("Question must not be empty or whitespace-only.")

    # ── 2. Load config ───────────────────────────────────────────────────
    if config is None:
        config = get_rag_config()

    k = top_k if top_k is not None else config.top_k

    print("\n" + "=" * 60)
    print("SEARCH: EVIDENCE RETRIEVAL (STAGES 5+6)")
    print("=" * 60)
    print(f'[*] Question      : "{cleaned}"')
    print(f"[*] Embed Model   : {config.embed_model}")
    print(f"[*] Top-K         : {k}")

    # ── 3. Embed the question ────────────────────────────────────────────
    try:
        query_vector = embed_text(
            text=cleaned,
            config=config,
            input_type="query",
        )
    except EmbeddingError as emb_err:
        raise SearchError(f"Failed to embed the question: {emb_err}") from None

    # Validate vector dimension before querying
    if len(query_vector) != EXPECTED_DIMENSION:
        raise SearchError(
            f"Query vector has {len(query_vector)} dimensions, "
            f"but the index expects {EXPECTED_DIMENSION}."
        )

    print(f"[OK] Query vector generated ({len(query_vector)} dims).")

    # ── 4. Query Pinecone / local vector store ───────────────────────────
    target = (
        f"Pinecone Index '{config.pinecone_index_name}'"
        if not config.use_local_vector_db
        else "Local Vector Store"
    )
    print(f"[*] Searching     : {target}")

    try:
        raw_matches = query_vector_db(
            query_vector=query_vector,
            top_k=k,
            config=config,
        )
    except VectorStoreError as vs_err:
        raise SearchError(f"Vector store query failed: {vs_err}") from None

    # ── 5. Handle empty results ──────────────────────────────────────────
    if not raw_matches:
        print("[!] No matching evidence found.")
        print("    Possible causes:")
        print("    - The index is empty (run the indexing pipeline first)")
        print("    - The question is unrelated to any indexed content")
        print("    - The index name or credentials are misconfigured")
        return []

    # ── 6. Validate and enrich each result ───────────────────────────────
    results: List[Dict[str, Any]] = []
    for rank_idx, raw_match in enumerate(raw_matches):
        enriched = _validate_and_enrich(raw_match, rank=rank_idx + 1)
        results.append(enriched)

    # ── 7. Print ranked evidence table ───────────────────────────────────
    print(f"\n[OK] Retrieved {len(results)} evidence chunks:\n")
    for r in results:
        score_pct = round(r["score"] * 100, 2)
        text_preview = r["text"].replace("\n", " ")
        if len(text_preview) > 80:
            text_preview = text_preview[:80] + "..."
        source_label = r["source_file"] or "unknown"
        chunk_label = f"chunk {r['chunk_index']}" if r["chunk_index"] is not None else "—"

        print(f"  #{r['rank']} [{r['relevance']:>8}] score={r['score']:.4f} ({score_pct}%)")
        print(f"     ID     : {r['id']}")
        print(f"     Source : {source_label} / {chunk_label}")
        print(f'     Text   : "{text_preview}"')
        print()

    high_count = sum(1 for r in results if r["relevance"] == "HIGH")
    mod_count = sum(1 for r in results if r["relevance"] == "MODERATE")
    print(f"[*] Quality Summary: {high_count} HIGH, {mod_count} MODERATE, "
          f"{len(results) - high_count - mod_count} LOW/NOISE out of {len(results)} results.")

    return results
