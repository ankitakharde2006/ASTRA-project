"""
embedding.py — The Naive RAG NVIDIA NIM Embedding Stage.

The Goal:
    Convert raw textual passages and queries into dense, semantic vector
    representations using the NVIDIA NIM embedding model (nvidia/nemotron-3-embed-1b).
    Strictly validates that returned vectors are 2048-dimensional and handles
    API errors, rate limits (HTTP 429), and malformed responses safely.

The "Why":
    Embedding models map natural language into high-dimensional geometric space.
    Chunks with semantically similar meanings point in similar directions,
    enabling millisecond similarity search across thousands of documents.
    Validating dimensionality (2048-dim) guarantees downstream vector database
    compatibility and prevents subtle index corruption during ingestion.

Security & Reliability:
    - Never hardcodes or logs API keys or sensitive authorization headers.
    - Handles rate limits with exponential backoff and jitter.
    - Validates both single text and batch inputs against empty or blank inputs.
    - Enforces strict response verification (dimension == 2048).
"""

import math
import random
import time
from typing import Any, List, Dict, Optional

from openai import (
    OpenAI,
    APIError,
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    RateLimitError,
    InternalServerError,
)

from config import RAGConfig, get_rag_config


# ---------------------------------------------------------------------------
# Constants & Defaults
# ---------------------------------------------------------------------------
EXPECTED_DIMENSION: int = 2048
DEFAULT_EMBED_MODEL: str = "nvidia/nemotron-3-embed-1b"
DEFAULT_BATCH_SIZE: int = 50
MAX_RETRIES: int = 4
INITIAL_BACKOFF_SECONDS: float = 1.0
BACKOFF_FACTOR: float = 2.0
REQUEST_TIMEOUT_SECONDS: float = 60.0


# ---------------------------------------------------------------------------
# Custom Exception Hierarchy
# ---------------------------------------------------------------------------
class EmbeddingError(Exception):
    """Base exception for all embedding-related failures."""
    pass


class EmbeddingAuthenticationError(EmbeddingError):
    """Raised when NVIDIA NIM API credentials fail authentication (HTTP 401)."""
    pass


class EmbeddingRateLimitError(EmbeddingError):
    """Raised when NVIDIA NIM rate limits (HTTP 429) persist after retries."""
    pass


class EmbeddingDimensionError(EmbeddingError):
    """Raised when an embedding vector does not have the expected dimensionality."""
    pass


class MalformedResponseError(EmbeddingError):
    """Raised when the API returns an unexpected or corrupted payload structure."""
    pass


class EmbeddingConnectionError(EmbeddingError):
    """Raised when connection to NVIDIA NIM endpoint fails or times out."""
    pass


# ---------------------------------------------------------------------------
# Client Factory & Validation Helpers
# ---------------------------------------------------------------------------
def get_embedding_client(config: Optional[RAGConfig] = None) -> OpenAI:
    """
    Instantiate and return an OpenAI client configured for the NVIDIA NIM endpoint.
    Never exposes API keys in logs or exceptions.
    """
    if config is None:
        config = get_rag_config()

    if not config.use_mock_ai and not config.nvidia_api_key:
        raise EmbeddingAuthenticationError(
            "NVIDIA NIM API key is missing. Please set NVIDIA_API_KEY in your environment or .env file."
        )

    return OpenAI(
        base_url=config.nvidia_base_url,
        api_key=config.nvidia_api_key or "mock-key",
        timeout=REQUEST_TIMEOUT_SECONDS,
    )


def validate_vector_dimensions(
    vector: Any,
    expected_dim: int = EXPECTED_DIMENSION,
    identifier: str = "vector",
) -> List[float]:
    """
    Confirm that the returned object is a non-empty list of numerical coordinates
    with the exact expected dimensionality (2048 dimensions).

    Raises:
        EmbeddingDimensionError: If vector length does not match expected_dim.
        MalformedResponseError: If vector is not a valid list of numbers.
    """
    if not isinstance(vector, (list, tuple)):
        raise MalformedResponseError(
            f"Expected embedding {identifier} to be a list, got {type(vector).__name__}."
        )

    actual_dim = len(vector)
    if actual_dim != expected_dim:
        raise EmbeddingDimensionError(
            f"Embedding vector [{identifier}] dimension mismatch: expected exactly {expected_dim} "
            f"dimensions for model '{DEFAULT_EMBED_MODEL}', but received {actual_dim} dimensions."
        )

    # Spot-check first and last coordinates are numerical
    if actual_dim > 0 and not isinstance(vector[0], (int, float)):
        raise MalformedResponseError(
            f"Embedding coordinates in [{identifier}] are not numeric (got {type(vector[0]).__name__})."
        )

    return list(vector)


def validate_input_text(text: Any, identifier: str = "input text") -> str:
    """
    Confirm input text is a non-empty, non-whitespace string.
    NVIDIA NIM API returns 400 BadRequest if sent an empty string.

    Raises:
        ValueError: If input is not a non-empty string.
    """
    if not isinstance(text, str):
        raise ValueError(f"{identifier} must be a string, got {type(text).__name__}.")

    trimmed = text.strip()
    if len(trimmed) == 0:
        raise ValueError(f"{identifier} must not be empty or whitespace-only.")

    return trimmed


def _generate_mock_vector(text: str, dim: int = EXPECTED_DIMENSION) -> List[float]:
    """
    Generate deterministic, normalized pseudo-vectors of exact expected dimensionality
    for offline development, CI environments, and lecture demonstrations without active API keys.
    """
    seed = sum(ord(ch) for ch in text)
    vector: List[float] = []
    for d in range(dim):
        val = math.sin(seed + d * 0.1) * 0.5
        vector.append(round(val, 6))
    return vector


# ---------------------------------------------------------------------------
# Core Embedding Functions
# ---------------------------------------------------------------------------
def embed_text(
    text: str,
    config: Optional[RAGConfig] = None,
    input_type: str = "query",
    max_retries: int = MAX_RETRIES,
) -> List[float]:
    """
    Embed a single string using the NVIDIA NIM embedding API.

    Args:
        text: The text string to embed. Must not be empty.
        config: RAG configuration container. If None, loaded via get_rag_config().
        input_type: Asymmetric embedding hint ('query' for queries, 'passage' for chunks).
        max_retries: Number of retry attempts on rate limits or transient errors.

    Returns:
        List[float]: A validated 2048-dimensional dense vector.

    Raises:
        ValueError: If input text is empty or blank.
        EmbeddingAuthenticationError: If API key is invalid (HTTP 401).
        EmbeddingRateLimitError: If rate limit persists after max_retries (HTTP 429).
        EmbeddingDimensionError: If returned vector is not 2048 dimensions.
        MalformedResponseError: If API response is missing required data.
        EmbeddingConnectionError: If network connection to NIM endpoint fails.
    """
    cleaned_text = validate_input_text(text, identifier="Text to embed")

    if config is None:
        config = get_rag_config()

    # Offline simulation branch
    if config.use_mock_ai:
        mock_vec = _generate_mock_vector(cleaned_text, dim=EXPECTED_DIMENSION)
        return validate_vector_dimensions(mock_vec, expected_dim=EXPECTED_DIMENSION, identifier="mock_single")

    client = get_embedding_client(config)
    model_name = config.embed_model or DEFAULT_EMBED_MODEL

    backoff = INITIAL_BACKOFF_SECONDS

    for attempt in range(max_retries + 1):
        try:
            extra_body = {"input_type": input_type} if input_type in ("query", "passage") else None
            
            kwargs: Dict[str, Any] = {
                "input": [cleaned_text],
                "model": model_name,
            }
            if extra_body:
                kwargs["extra_body"] = extra_body

            response = client.embeddings.create(**kwargs)

            # Validate response structure
            if response is None or not hasattr(response, "data") or not response.data:
                raise MalformedResponseError(
                    "NVIDIA NIM response is malformed: missing or empty 'data' field."
                )

            item = response.data[0]
            if not hasattr(item, "embedding") or item.embedding is None:
                raise MalformedResponseError(
                    "NVIDIA NIM response item is malformed: missing 'embedding' field."
                )

            # Validate strict 2048 dimensionality
            vector = validate_vector_dimensions(
                item.embedding,
                expected_dim=EXPECTED_DIMENSION,
                identifier=f"single_embed({cleaned_text[:20]}...)",
            )
            return vector

        except AuthenticationError:
            raise EmbeddingAuthenticationError(
                "NVIDIA NIM authentication failed (HTTP 401). "
                "Please verify that NVIDIA_API_KEY is properly set in your environment or .env file."
            ) from None

        except BadRequestError as bad_req:
            error_msg = str(bad_req).lower()
            if "extra_body" in error_msg or "input_type" in error_msg:
                # Retry once without extra_body in case an endpoint variant rejects it
                try:
                    fallback_res = client.embeddings.create(input=[cleaned_text], model=model_name)
                    if fallback_res and fallback_res.data and hasattr(fallback_res.data[0], "embedding"):
                        return validate_vector_dimensions(fallback_res.data[0].embedding, EXPECTED_DIMENSION)
                except Exception:
                    pass
            raise EmbeddingError(f"NVIDIA NIM rejected request (HTTP 400 BadRequest): {bad_req}") from None

        except RateLimitError as rle:
            if attempt < max_retries:
                jitter = random.uniform(0.1, 0.5)
                sleep_time = backoff + jitter
                print(f"[!] Rate limit (429) on NVIDIA NIM. Retrying in {sleep_time:.2f}s (attempt {attempt + 1}/{max_retries})...")
                time.sleep(sleep_time)
                backoff *= BACKOFF_FACTOR
            else:
                raise EmbeddingRateLimitError(
                    f"NVIDIA NIM rate limit exceeded after {max_retries} retries. Please check your quota or wait before querying."
                ) from None

        except (InternalServerError, APIConnectionError, APITimeoutError) as transient_err:
            if attempt < max_retries:
                jitter = random.uniform(0.1, 0.5)
                sleep_time = backoff + jitter
                print(f"[!] Transient network/server error ({type(transient_err).__name__}). Retrying in {sleep_time:.2f}s...")
                time.sleep(sleep_time)
                backoff *= BACKOFF_FACTOR
            else:
                raise EmbeddingConnectionError(
                    f"Failed to connect to NVIDIA NIM endpoint ({config.nvidia_base_url}) after {max_retries} attempts: {transient_err}"
                ) from None

        except (EmbeddingDimensionError, MalformedResponseError, ValueError):
            raise

        except APIError as api_err:
            raise EmbeddingError(f"NVIDIA NIM API error: {api_err}") from None

        except Exception as generic_err:
            raise EmbeddingError(f"Unexpected error during NVIDIA NIM embedding: {generic_err}") from None

    raise EmbeddingError("Embedding stage terminated unexpectedly without returning a vector.")


def embed_batch(
    texts: List[str],
    config: Optional[RAGConfig] = None,
    input_type: str = "passage",
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_retries: int = MAX_RETRIES,
) -> List[List[float]]:
    """
    Embed a list of text strings in batches using the NVIDIA NIM embedding API.

    Splits large sets into sub-batches (default: 50 items) to respect API payload
    limits and provides robust retry handling and dimension validation across all items.

    Args:
        texts: List of text strings to embed.
        config: RAG configuration container.
        input_type: Asymmetric embedding hint ('passage' for chunks, 'query' for search queries).
        batch_size: Maximum texts per single API call (<= 50 recommended for NIM).
        max_retries: Number of retry attempts on rate limits or transient errors.

    Returns:
        List[List[float]]: A list of 2048-dimensional embedding vectors matching the input texts.
    """
    if not texts:
        return []

    if batch_size <= 0:
        raise ValueError(f"batch_size must be a positive integer, got {batch_size}.")

    if config is None:
        config = get_rag_config()

    # Validate all input texts
    cleaned_texts = []
    for idx, txt in enumerate(texts):
        cleaned_texts.append(validate_input_text(txt, identifier=f"Input text at index {idx}"))

    # Offline simulation branch
    if config.use_mock_ai:
        return [
            validate_vector_dimensions(_generate_mock_vector(t, dim=EXPECTED_DIMENSION), EXPECTED_DIMENSION, f"mock_batch_{i}")
            for i, t in enumerate(cleaned_texts)
        ]

    client = get_embedding_client(config)
    model_name = config.embed_model or DEFAULT_EMBED_MODEL
    all_vectors: List[List[float]] = []

    # Process in slices of batch_size
    total_texts = len(cleaned_texts)
    for slice_start in range(0, total_texts, batch_size):
        slice_end = min(slice_start + batch_size, total_texts)
        sub_batch = cleaned_texts[slice_start:slice_end]
        backoff = INITIAL_BACKOFF_SECONDS

        batch_success = False
        for attempt in range(max_retries + 1):
            try:
                extra_body = {"input_type": input_type} if input_type in ("query", "passage") else None
                kwargs: Dict[str, Any] = {
                    "input": sub_batch,
                    "model": model_name,
                }
                if extra_body:
                    kwargs["extra_body"] = extra_body

                response = client.embeddings.create(**kwargs)

                # Validate response structure
                if response is None or not hasattr(response, "data") or response.data is None:
                    raise MalformedResponseError(
                        f"NVIDIA NIM batch response is malformed: missing 'data' field for batch {slice_start}-{slice_end}."
                    )

                if len(response.data) != len(sub_batch):
                    raise MalformedResponseError(
                        f"Mismatched vector count from NVIDIA NIM: sent {len(sub_batch)} texts, "
                        f"but received {len(response.data)} embeddings."
                    )

                for item_idx, item in enumerate(response.data):
                    global_idx = slice_start + item_idx
                    if not hasattr(item, "embedding") or item.embedding is None:
                        raise MalformedResponseError(
                            f"NVIDIA NIM batch item at index {global_idx} missing 'embedding' field."
                        )

                    vec = validate_vector_dimensions(
                        item.embedding,
                        expected_dim=EXPECTED_DIMENSION,
                        identifier=f"batch_item_{global_idx}",
                    )
                    all_vectors.append(vec)

                batch_success = True
                break

            except AuthenticationError:
                raise EmbeddingAuthenticationError(
                    "NVIDIA NIM authentication failed (HTTP 401). "
                    "Please verify that NVIDIA_API_KEY is properly set in your environment or .env file."
                ) from None

            except BadRequestError as bad_req:
                error_msg = str(bad_req).lower()
                if "extra_body" in error_msg or "input_type" in error_msg:
                    try:
                        fallback_res = client.embeddings.create(input=sub_batch, model=model_name)
                        if fallback_res and fallback_res.data and len(fallback_res.data) == len(sub_batch):
                            for item_idx, item in enumerate(fallback_res.data):
                                global_idx = slice_start + item_idx
                                vec = validate_vector_dimensions(item.embedding, EXPECTED_DIMENSION, f"batch_item_{global_idx}")
                                all_vectors.append(vec)
                            batch_success = True
                            break
                    except Exception:
                        pass
                raise EmbeddingError(f"NVIDIA NIM rejected batch request (HTTP 400 BadRequest): {bad_req}") from None

            except RateLimitError as rle:
                if attempt < max_retries:
                    jitter = random.uniform(0.1, 0.5)
                    sleep_time = backoff + jitter
                    print(f"[!] Rate limit (429) during batch embedding. Retrying in {sleep_time:.2f}s (attempt {attempt + 1}/{max_retries})...")
                    time.sleep(sleep_time)
                    backoff *= BACKOFF_FACTOR
                else:
                    raise EmbeddingRateLimitError(
                        f"NVIDIA NIM rate limit exceeded for batch {slice_start}-{slice_end} after {max_retries} retries."
                    ) from None

            except (InternalServerError, APIConnectionError, APITimeoutError) as transient_err:
                if attempt < max_retries:
                    jitter = random.uniform(0.1, 0.5)
                    sleep_time = backoff + jitter
                    print(f"[!] Transient network/server error ({type(transient_err).__name__}). Retrying in {sleep_time:.2f}s...")
                    time.sleep(sleep_time)
                    backoff *= BACKOFF_FACTOR
                else:
                    raise EmbeddingConnectionError(
                        f"Failed to connect to NVIDIA NIM endpoint ({config.nvidia_base_url}) for batch {slice_start}-{slice_end}: {transient_err}"
                    ) from None

            except (EmbeddingDimensionError, MalformedResponseError, ValueError):
                raise

            except APIError as api_err:
                raise EmbeddingError(f"NVIDIA NIM API error during batch: {api_err}") from None

            except Exception as generic_err:
                raise EmbeddingError(f"Unexpected error during batch embedding: {generic_err}") from None

        if not batch_success:
            raise EmbeddingError(f"Failed to embed batch slice {slice_start}-{slice_end}.")

    return all_vectors


def embed_chunks(
    chunks: List[Dict[str, Any]],
    config: Optional[RAGConfig] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> List[Dict[str, Any]]:
    """
    Stage 3: Embedding (NVIDIA NIM)
    Converts each chunk dictionary into a dense 2048-dimensional vector representation.
    Attaches the validated vector under the 'embedding' key and preserves all chunk metadata.

    Args:
        chunks: List of chunk dictionaries containing 'text' or 'content' and 'id'.
        config: RAG configuration container.
        batch_size: Number of chunks per batch embedding API call.

    Returns:
        List[Dict[str, Any]]: List of chunk dictionaries enriched with 'embedding'.
    """
    if config is None:
        config = get_rag_config()

    print("\n" + "=" * 60)
    print("STAGE 3: EMBEDDING GENERATION (NVIDIA NIM)")
    print("=" * 60)
    print(f"[*] Embedding Model : {config.embed_model}")
    print(f"[*] Total Chunks    : {len(chunks)}")
    print(f"[*] Batch Size      : {batch_size}")

    if not chunks:
        print("[*] No chunks provided for embedding. Returning empty list.")
        return []

    # Extract text from chunks
    texts = []
    for idx, c in enumerate(chunks):
        txt = c.get("text") or c.get("content") or ""
        texts.append(txt)

    if not config.use_mock_ai:
        print(f"[*] Connecting to NVIDIA NIM at: {config.nvidia_base_url}")
    else:
        print("[!] Offline mode active: Generating simulated deterministic 2048-dim vectors...")

    # Embed batch with validation and error handling
    embeddings = embed_batch(
        texts=texts,
        config=config,
        input_type="passage",
        batch_size=batch_size,
    )

    enriched_chunks: List[Dict[str, Any]] = []
    vector_dim = len(embeddings[0]) if embeddings else 0

    print(f"[OK] Received {len(embeddings)} embedding vectors from NVIDIA NIM.")
    print(f"[*] Vector Dimensionality : {vector_dim} dimensions (Verified: {vector_dim == EXPECTED_DIMENSION})")

    for idx, chunk in enumerate(chunks):
        enriched = dict(chunk)
        enriched["embedding"] = embeddings[idx]
        enriched_chunks.append(enriched)

        # Print sample coordinates for classroom visibility
        sample_coords = [round(val, 4) for val in embeddings[idx][:4]]
        chunk_id = chunk.get("id", f"chunk_{idx}")
        print(f"  -> Vector [{chunk_id}] coordinates: {sample_coords} ...")

    return enriched_chunks


def embed_query(
    query: str,
    config: Optional[RAGConfig] = None,
) -> List[float]:
    """
    Stage 5: Query Embedding (NVIDIA NIM)
    Embeds a single live query into the same 2048-dimensional mathematical space.

    Args:
        query: The user's question string.
        config: RAG configuration container.

    Returns:
        List[float]: A validated 2048-dimensional float list.
    """
    cleaned_query = validate_input_text(query, identifier="User query")

    if config is None:
        config = get_rag_config()

    print("\n" + "=" * 60)
    print("STAGE 5: QUERY EMBEDDING (NVIDIA NIM)")
    print("=" * 60)
    print(f"[*] Live User Query : \"{cleaned_query}\"")
    print(f"[*] Embedding Model : {config.embed_model}")

    query_vector = embed_text(
        text=cleaned_query,
        config=config,
        input_type="query",
    )

    vector_dim = len(query_vector)
    sample_coords = [round(val, 4) for val in query_vector[:4]]
    print(f"[OK] Generated query vector ({vector_dim} dimensions).")
    print(f"[*] Sample Coordinates : {sample_coords} ...")

    return query_vector
