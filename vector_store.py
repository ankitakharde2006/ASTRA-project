"""
vector_store.py — Vector Database Management (Pinecone with Local Fallback).

The Goal:
    Provide an interface to initialize, upsert, and query vector embeddings
    using Pinecone Serverless. If no Pinecone key is configured, it falls back
    to a transparent local JSON vector store so live demos never fail.

The "Why":
    In production, teams use managed vector databases like Pinecone to perform
    millisecond-level Approximate Nearest Neighbor (ANN) searches across millions
    of vectors. However, during a live classroom lecture or workshop, API keys
    can expire, or the venue Wi-Fi can drop. Providing a local fallback guarantees
    pedagogical continuity without altering the RAG pipeline interface.

Stage 4 Enhancements:
    - Validates index dimension (2048) and metric (cosine) before upserting.
    - Uses deterministic chunk IDs so repeated indexing of identical source files
      overwrites existing records rather than creating duplicates.
    - Batch-upserts in slices of up to 100 vectors (Pinecone recommended limit).
    - Stores chunk text and source metadata for later retrieval.
    - Handles Pinecone service errors, authentication failures, and invalid
      configuration with clear diagnostics.
    - Never prints or logs API keys.
"""

import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from pinecone import Pinecone, ServerlessSpec
from config import RAGConfig


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LOCAL_DB_FILE = Path(__file__).resolve().parent / "data" / "local_vector_db.json"
EXPECTED_DIMENSION: int = 2048
EXPECTED_METRIC: str = "cosine"
PINECONE_UPSERT_BATCH_SIZE: int = 100
INDEX_READY_POLL_INTERVAL: float = 2.0
INDEX_READY_MAX_WAIT: float = 120.0


# ---------------------------------------------------------------------------
# Custom Exceptions
# ---------------------------------------------------------------------------
class VectorStoreError(Exception):
    """Base exception for all vector store failures."""
    pass


class VectorStoreConfigError(VectorStoreError):
    """Raised when Pinecone configuration is missing or invalid."""
    pass


class VectorStoreDimensionError(VectorStoreError):
    """Raised when the vector dimension does not match the index dimension."""
    pass


class VectorStoreConnectionError(VectorStoreError):
    """Raised when Pinecone API connectivity fails."""
    pass


# ---------------------------------------------------------------------------
# Cosine Similarity (Local Fallback)
# ---------------------------------------------------------------------------
def calculate_cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Calculate cosine similarity between two vectors using basic math:
    Cosine(A, B) = dot_product(A, B) / (norm(A) * norm(B))
    
    [TEACHING NOTE]
    Cosine similarity measures the angle between two embedding vectors in
    high-dimensional space. A score of 1.0 means identical direction/meaning,
    while 0.0 means completely orthogonal (unrelated).
    """
    if len(vec_a) != len(vec_b):
        return 0.0

    dot_product = 0.0
    norm_a_sq = 0.0
    norm_b_sq = 0.0

    index = 0
    while index < len(vec_a):
        a_val = vec_a[index]
        b_val = vec_b[index]
        dot_product = dot_product + (a_val * b_val)
        norm_a_sq = norm_a_sq + (a_val * a_val)
        norm_b_sq = norm_b_sq + (b_val * b_val)
        index = index + 1

    if norm_a_sq <= 0.0 or norm_b_sq <= 0.0:
        return 0.0

    norm_a = math.sqrt(norm_a_sq)
    norm_b = math.sqrt(norm_b_sq)
    return dot_product / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# Pinecone Client & Index Management
# ---------------------------------------------------------------------------
def get_pinecone_client(api_key: str) -> Pinecone:
    """
    Instantiate and return the Pinecone client (SDK v10).
    Never logs or prints the API key.

    Raises:
        VectorStoreConfigError: If api_key is empty or blank.
    """
    if not api_key or not api_key.strip():
        raise VectorStoreConfigError(
            "PINECONE_API_KEY is missing or empty. "
            "Please set it in your .env file or environment variables."
        )
    return Pinecone(api_key=api_key)


def validate_index_compatibility(
    client: Pinecone,
    index_name: str,
    expected_dimension: int = EXPECTED_DIMENSION,
    expected_metric: str = EXPECTED_METRIC,
) -> Dict[str, Any]:
    """
    Verify that an existing Pinecone index matches the expected dimension and metric.
    Returns the index description dict on success.

    Raises:
        VectorStoreConfigError: If the index does not exist.
        VectorStoreDimensionError: If dimension or metric does not match.
    """
    try:
        idx_info = client.describe_index(index_name)
    except Exception as err:
        raise VectorStoreConfigError(
            f"Cannot describe Pinecone index '{index_name}': {err}"
        ) from None

    # Extract dimension and metric from the index description
    idx_dict = idx_info.to_dict() if hasattr(idx_info, "to_dict") else {}
    actual_dim = None
    actual_metric = None

    # Direct attributes (standard dense indexes)
    if hasattr(idx_info, "dimension") and idx_info.dimension is not None:
        try:
            actual_dim = idx_info.dimension
        except AttributeError:
            pass
    if hasattr(idx_info, "metric") and idx_info.metric is not None:
        try:
            actual_metric = idx_info.metric
        except AttributeError:
            pass

    # Fallback: parse from schema fields for dense_vector type
    if actual_dim is None:
        schema = idx_dict.get("schema", {})
        fields = schema.get("fields", {})
        for field_name, field_info in fields.items():
            ftype = field_info.get("type", "")
            if ftype == "dense_vector":
                actual_dim = field_info.get("dimension")
                actual_metric = field_info.get("metric", actual_metric)
                break

    # Validate dimension
    if actual_dim is not None and actual_dim != expected_dimension:
        raise VectorStoreDimensionError(
            f"Pinecone index '{index_name}' has dimension {actual_dim}, "
            f"but the embedding model produces {expected_dimension}-dimensional vectors. "
            f"These must match exactly."
        )

    # Validate metric
    if actual_metric is not None and actual_metric != expected_metric:
        raise VectorStoreDimensionError(
            f"Pinecone index '{index_name}' uses metric '{actual_metric}', "
            f"but this pipeline requires '{expected_metric}'."
        )

    return idx_dict


def ensure_pinecone_index_exists(
    client: Pinecone,
    index_name: str,
    dimension: int = EXPECTED_DIMENSION,
    metric: str = EXPECTED_METRIC,
    cloud: str = "aws",
    region: str = "us-east-1",
) -> None:
    """
    Check if the Pinecone index exists; if not, create a Serverless index
    with the correct dimension and metric, then wait until it is ready.

    If the index already exists, validate that its dimension and metric
    are compatible with our 2048-dim cosine embeddings.

    Raises:
        VectorStoreDimensionError: If an existing index has wrong dimension/metric.
        VectorStoreConnectionError: If index creation or readiness check fails.
    """
    existing_indexes = client.list_indexes()
    index_names = []
    for item in existing_indexes:
        index_names.append(item.name)

    if index_name in index_names:
        print(f"[OK] Connected to existing Pinecone Index: '{index_name}'.")
        validate_index_compatibility(client, index_name, dimension, metric)
        return

    # Create the index
    print(f"[*] Pinecone Index '{index_name}' not found. Creating serverless index...")
    try:
        client.create_index(
            name=index_name,
            dimension=dimension,
            metric=metric,
            spec=ServerlessSpec(cloud=cloud, region=region),
        )
    except Exception as err:
        raise VectorStoreConnectionError(
            f"Failed to create Pinecone index '{index_name}': {err}"
        ) from None

    # Poll until the index is ready
    elapsed = 0.0
    while elapsed < INDEX_READY_MAX_WAIT:
        try:
            idx_info = client.describe_index(index_name)
            status = idx_info.status
            if hasattr(status, "ready") and status.ready:
                break
            if hasattr(status, "state") and status.state == "Ready":
                break
        except Exception:
            pass
        time.sleep(INDEX_READY_POLL_INTERVAL)
        elapsed += INDEX_READY_POLL_INTERVAL

    print(f"[OK] Pinecone Index '{index_name}' created successfully "
          f"(Dimension: {dimension}, Metric: {metric}).")


# ---------------------------------------------------------------------------
# Chunk ID Generation (Idempotent)
# ---------------------------------------------------------------------------
def make_chunk_id(chunk: Dict[str, Any]) -> str:
    """
    Produce a stable, deterministic ID for a chunk so that repeated indexing
    of the same source files overwrites (upserts) existing records rather than
    creating duplicates.

    Priority order:
        1. chunk["id"]           — already assigned by the chunking stage
        2. file_path#chunk_index — constructed from metadata
        3. "chunk_{index}"       — last resort fallback
    """
    if chunk.get("id"):
        return str(chunk["id"])

    file_path = chunk.get("file_path") or chunk.get("relative_path") or chunk.get("source_file") or ""
    chunk_index = chunk.get("chunk_index")
    if file_path and chunk_index is not None:
        return f"{file_path}#chunk_{chunk_index}"

    return f"chunk_{id(chunk)}"


def build_metadata(chunk: Dict[str, Any]) -> Dict[str, str]:
    """
    Build the metadata dict to store alongside the vector in Pinecone.
    Pinecone metadata values must be strings, numbers, booleans, or lists of strings.
    Stores the original chunk text and source information for later retrieval.
    """
    text = chunk.get("text") or chunk.get("content") or ""
    metadata: Dict[str, str] = {
        "text": text,
        "chunk_id": make_chunk_id(chunk),
    }

    # Preserve source file information if available
    source_file = chunk.get("file_path") or chunk.get("relative_path") or chunk.get("source_file")
    if source_file:
        metadata["source_file"] = str(source_file)

    chunk_index = chunk.get("chunk_index")
    if chunk_index is not None:
        metadata["chunk_index"] = str(chunk_index)

    return metadata


# ---------------------------------------------------------------------------
# Core Upsert
# ---------------------------------------------------------------------------
def upsert_to_vector_db(
    chunks_with_embeddings: list[dict],
    config: RAGConfig,
) -> int:
    """
    Stage 4: Store chunk embeddings and text metadata into Pinecone or local fallback.

    Behaviour:
        - Validates every vector has exactly 2048 dimensions before upserting.
        - Uses deterministic chunk IDs so repeated indexing is idempotent (upsert
          overwrites existing records with the same ID).
        - Batch-upserts in slices of 100 vectors per API call.
        - Stores 'text', 'chunk_id', 'source_file', and 'chunk_index' as metadata.
        - Falls back to local JSON storage if Pinecone is unavailable.

    Returns:
        int: Number of vectors successfully stored.

    Raises:
        VectorStoreDimensionError: If any embedding vector is not 2048 dimensions.
        VectorStoreConfigError: If Pinecone credentials are invalid.
    """
    count = len(chunks_with_embeddings)
    if count == 0:
        return 0

    # Pre-validate all vector dimensions before any network call
    for idx, item in enumerate(chunks_with_embeddings):
        embedding = item.get("embedding")
        if embedding is None:
            raise VectorStoreDimensionError(
                f"Chunk at index {idx} (id={item.get('id', '?')}) is missing the 'embedding' field."
            )
        dim = len(embedding)
        if dim != EXPECTED_DIMENSION:
            raise VectorStoreDimensionError(
                f"Chunk at index {idx} (id={item.get('id', '?')}) has {dim} dimensions, "
                f"but the Pinecone index requires exactly {EXPECTED_DIMENSION}."
            )

    # Branch 1: Use Pinecone Cloud
    if not config.use_local_vector_db:
        try:
            client = get_pinecone_client(config.pinecone_api_key)

            ensure_pinecone_index_exists(
                client=client,
                index_name=config.pinecone_index_name,
                dimension=EXPECTED_DIMENSION,
                metric=EXPECTED_METRIC,
                cloud=config.pinecone_cloud,
                region=config.pinecone_region,
            )

            index = client.Index(config.pinecone_index_name)

            # Build the full payload
            vectors_payload: List[Dict[str, Any]] = []
            for item in chunks_with_embeddings:
                vec_id = make_chunk_id(item)
                record = {
                    "id": vec_id,
                    "values": item["embedding"],
                    "metadata": build_metadata(item),
                }
                vectors_payload.append(record)

            # Batch upsert in slices of PINECONE_UPSERT_BATCH_SIZE
            total_upserted = 0
            for batch_start in range(0, len(vectors_payload), PINECONE_UPSERT_BATCH_SIZE):
                batch_end = min(batch_start + PINECONE_UPSERT_BATCH_SIZE, len(vectors_payload))
                batch_slice = vectors_payload[batch_start:batch_end]

                try:
                    upsert_response = index.upsert(vectors=batch_slice)
                    upserted = getattr(upsert_response, "upserted_count", len(batch_slice))
                    total_upserted += upserted
                    print(f"  -> Batch [{batch_start}:{batch_end}] upserted {upserted} vectors.")
                except Exception as batch_err:
                    print(f"[!] Error upserting batch [{batch_start}:{batch_end}]: {batch_err}")
                    raise VectorStoreConnectionError(
                        f"Pinecone upsert failed for batch [{batch_start}:{batch_end}]: {batch_err}"
                    ) from None

            print(f"[OK] Successfully upserted {total_upserted} vectors to "
                  f"Pinecone Index '{config.pinecone_index_name}'.")

            # Log idempotency note
            print(f"[*] Idempotent IDs: re-indexing the same source files will overwrite, not duplicate.")
            return total_upserted

        except (VectorStoreError,):
            raise
        except Exception as error:
            print(f"[!] Warning: Pinecone upsert encountered an error: {error}")
            print("[!] Falling back to Local Vector Database for live reliability.")

    # Branch 2: Local JSON Storage Fallback
    LOCAL_DB_FILE.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for item in chunks_with_embeddings:
        record = {
            "id": make_chunk_id(item),
            "embedding": item["embedding"],
            "metadata": build_metadata(item),
        }
        records.append(record)

    with open(LOCAL_DB_FILE, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    print(f"[OK] Saved {count} vectors to Local Vector Store ({LOCAL_DB_FILE.name}).")
    return count


# ---------------------------------------------------------------------------
# Core Query
# ---------------------------------------------------------------------------
def query_vector_db(
    query_vector: list[float],
    top_k: int,
    config: RAGConfig,
) -> list[dict]:
    """
    Retrieve top-k most similar chunks for a query vector from Pinecone or local fallback.
    Returns chunk records with 'id', 'score', and 'text' fields.
    """
    results = []

    # Branch 1: Query Pinecone Cloud
    if not config.use_local_vector_db:
        try:
            client = get_pinecone_client(config.pinecone_api_key)
            index = client.Index(config.pinecone_index_name)
            response = index.query(
                vector=query_vector,
                top_k=top_k,
                include_metadata=True,
            )

            for match in response.matches:
                meta = match.metadata or {}
                chunk_record = {
                    "id": match.id,
                    "score": float(match.score),
                    "text": meta.get("text", ""),
                }
                results.append(chunk_record)

            return results

        except Exception as error:
            print(f"[!] Warning: Pinecone query failed ({error}). Using local fallback.")

    # Branch 2: Local Vector Store Fallback
    if not LOCAL_DB_FILE.exists():
        print(f"[!] Local Vector Store file not found at {LOCAL_DB_FILE}. Please run indexing first.")
        return []

    with open(LOCAL_DB_FILE, "r", encoding="utf-8") as f:
        stored_records = json.load(f)

    scored_items = []
    for record in stored_records:
        candidate_vector = record["embedding"]
        similarity = calculate_cosine_similarity(query_vector, candidate_vector)
        item = {
            "id": record["id"],
            "score": similarity,
            "text": record["metadata"]["text"],
        }
        scored_items.append(item)

    # Sort descending by similarity score
    def get_item_score(x: dict) -> float:
        return x["score"]

    scored_items.sort(key=get_item_score, reverse=True)

    # Select top_k elements
    selected_items = []
    count = 0
    for entry in scored_items:
        if count < top_k:
            selected_items.append(entry)
            count = count + 1

    return selected_items


# ---------------------------------------------------------------------------
# Verification Helpers
# ---------------------------------------------------------------------------
def verify_pinecone_vectors(
    config: RAGConfig,
    sample_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Verify that vectors were successfully inserted into Pinecone and that
    stored metadata can be retrieved.

    Returns a summary dict with index stats and optionally fetched sample records.
    Never prints API keys.

    Args:
        config: RAG configuration.
        sample_ids: Optional list of chunk IDs to fetch and inspect.

    Returns:
        Dict with 'index_name', 'dimension', 'metric', 'total_vectors',
        'namespaces', and optionally 'fetched_samples'.
    """
    client = get_pinecone_client(config.pinecone_api_key)
    index = client.Index(config.pinecone_index_name)

    stats = index.describe_index_stats()
    result: Dict[str, Any] = {
        "index_name": config.pinecone_index_name,
        "dimension": stats.dimension,
        "metric": stats.metric,
        "total_vectors": stats.total_vector_count,
        "namespaces": stats.to_dict().get("namespaces", {}),
    }

    if sample_ids:
        fetched = index.fetch(ids=sample_ids)
        samples = {}
        for vid, vdata in fetched.vectors.items():
            samples[vid] = {
                "metadata": dict(vdata.metadata) if vdata.metadata else {},
                "values_length": len(vdata.values) if vdata.values else 0,
            }
        result["fetched_samples"] = samples

    return result
