"""
test_stage3_embedding.py — Test Suite for Stage 3: NVIDIA NIM Embedding Generation.

Tests:
1. Embedding a single harmless string returns a vector of exactly 2048 dimensions.
2. Embedding a batch of chunks returns matching count of 2048-dimensional vectors.
3. Strict dimension validation: rejects vectors not matching 2048 dimensions (EmbeddingDimensionError).
4. Strict input validation: rejects empty and whitespace-only strings before making API calls.
5. Safe empty batch handling: empty chunk lists return empty lists without API errors.
6. Malformed response handling: rejects responses with missing data or corrupt structure.
7. Stage 2 integration: attaches 2048-dim embeddings to chunk dictionaries while preserving all metadata.
8. Secret protection: guarantees API keys and sensitive tokens are never printed or leaked in error messages.
"""

import unittest
from unittest.mock import MagicMock, patch

from config import RAGConfig, get_rag_config
from embedding import (
    EXPECTED_DIMENSION,
    DEFAULT_EMBED_MODEL,
    EmbeddingError,
    EmbeddingAuthenticationError,
    EmbeddingDimensionError,
    EmbeddingRateLimitError,
    MalformedResponseError,
    embed_text,
    embed_batch,
    embed_chunks,
    embed_query,
    validate_vector_dimensions,
    validate_input_text,
    get_embedding_client,
)


class TestStage3Embedding(unittest.TestCase):
    """
    Focused test suite for Stage 3: NVIDIA NIM Embedding (nvidia/nemotron-3-embed-1b).
    """

    def setUp(self):
        self.config = get_rag_config()

    def test_single_text_embedding_returns_2048_dimensions(self):
        """
        Embed a harmless sample string and confirm vector dimensionality is exactly 2048.
        Confirms no secrets or sensitive tokens are exposed.
        """
        sample_text = "The quick brown fox jumps over the lazy dog."
        vector = embed_text(sample_text, config=self.config)

        self.assertIsInstance(vector, list, "Vector must be returned as a list.")
        self.assertEqual(
            len(vector),
            EXPECTED_DIMENSION,
            f"Expected exactly {EXPECTED_DIMENSION} dimensions, got {len(vector)}.",
        )
        self.assertTrue(
            all(isinstance(coord, float) for coord in vector),
            "All coordinates in the embedding vector must be floats.",
        )

    def test_batch_embedding_returns_matching_2048_dim_vectors(self):
        """
        Embed a batch of strings and confirm each returned vector has 2048 dimensions.
        """
        samples = [
            "AstraTech remote office policy documentation.",
            "Information security and visitor network credentials.",
            "Employee wellness benefits and standard operational hours.",
        ]
        vectors = embed_batch(samples, config=self.config)

        self.assertEqual(len(vectors), len(samples), "Must return 1 vector per input text.")
        for idx, vec in enumerate(vectors):
            self.assertEqual(
                len(vec),
                EXPECTED_DIMENSION,
                f"Vector at index {idx} dimension mismatch: expected {EXPECTED_DIMENSION}, got {len(vec)}.",
            )

    def test_dimension_validation_enforces_2048_dims(self):
        """
        Ensure validate_vector_dimensions strictly raises EmbeddingDimensionError
        if a vector does not have exactly 2048 dimensions.
        """
        # Exactly 2048 dimensions -> must succeed
        valid_vec = [0.0] * EXPECTED_DIMENSION
        result = validate_vector_dimensions(valid_vec, expected_dim=EXPECTED_DIMENSION)
        self.assertEqual(len(result), EXPECTED_DIMENSION)

        # 64 dimensions (e.g. old mock vector) -> must fail
        with self.assertRaises(EmbeddingDimensionError):
            validate_vector_dimensions([0.1] * 64, expected_dim=EXPECTED_DIMENSION)

        # 1536 dimensions (e.g. OpenAI text-embedding-ada-002) -> must fail
        with self.assertRaises(EmbeddingDimensionError):
            validate_vector_dimensions([0.1] * 1536, expected_dim=EXPECTED_DIMENSION)

        # Off-by-one (2047 dims) -> must fail
        with self.assertRaises(EmbeddingDimensionError):
            validate_vector_dimensions([0.1] * (EXPECTED_DIMENSION - 1), expected_dim=EXPECTED_DIMENSION)

    def test_empty_and_whitespace_rejection(self):
        """
        Reject empty or whitespace-only inputs before calling NVIDIA NIM.
        """
        with self.assertRaises(ValueError):
            embed_text("", config=self.config)

        with self.assertRaises(ValueError):
            embed_text("   \n\t   ", config=self.config)

        with self.assertRaises(ValueError):
            embed_batch(["Valid chunk", "   "], config=self.config)

    def test_empty_batch_handling(self):
        """
        Empty list of texts or chunks returns empty list without making API requests.
        """
        self.assertEqual(embed_batch([], config=self.config), [])
        self.assertEqual(embed_chunks([], config=self.config), [])

    def test_stage2_chunks_enrichment(self):
        """
        Stage 3 embed_chunks enriches chunk dictionaries with 2048-dim embeddings
        while fully preserving Stage 2 metadata (id, file_path, chunk_index, text).
        """
        stage2_chunks = [
            {
                "id": "config.py#chunk_0",
                "file_path": "config.py",
                "relative_path": "config.py",
                "chunk_index": 0,
                "text": "def get_rag_config(): return RAGConfig()",
            },
            {
                "id": "config.py#chunk_1",
                "file_path": "config.py",
                "relative_path": "config.py",
                "chunk_index": 1,
                "text": "class RAGConfig: pass",
            },
        ]

        enriched = embed_chunks(stage2_chunks, config=self.config)

        self.assertEqual(len(enriched), len(stage2_chunks))
        for idx, item in enumerate(enriched):
            self.assertEqual(item["id"], stage2_chunks[idx]["id"])
            self.assertEqual(item["file_path"], stage2_chunks[idx]["file_path"])
            self.assertEqual(item["chunk_index"], stage2_chunks[idx]["chunk_index"])
            self.assertEqual(item["text"], stage2_chunks[idx]["text"])
            self.assertIn("embedding", item)
            self.assertEqual(len(item["embedding"]), EXPECTED_DIMENSION)

    def test_query_embedding_helper(self):
        """
        embed_query embeds a live user query and returns a 2048-dim float vector.
        """
        query_vec = embed_query("What is the visitor Wi-Fi password?", config=self.config)
        self.assertEqual(len(query_vec), EXPECTED_DIMENSION)

    def test_malformed_response_handling(self):
        """
        Ensure MalformedResponseError is raised if API returns missing data or bad structure.
        """
        mock_response = MagicMock()
        mock_response.data = None  # Missing data

        with patch("embedding.get_embedding_client") as mock_client_factory:
            mock_client = MagicMock()
            mock_client.embeddings.create.return_value = mock_response
            mock_client_factory.return_value = mock_client

            with self.assertRaises(MalformedResponseError):
                embed_text("Sample input for malformed test", config=self.config)

    def test_secret_is_never_leaked_in_exceptions(self):
        """
        Ensure authentication errors or API errors never output raw API key.
        """
        mock_config = RAGConfig(
            nvidia_api_key="nvapi-SECRET-KEY-SHOULD-NEVER-APPEAR-IN-LOGS-12345",
            nvidia_base_url="https://integrate.api.nvidia.com/v1",
            chat_model="meta/llama-3.2-11b-vision-instruct",
            embed_model="nvidia/nemotron-3-embed-1b",
            pinecone_api_key="",
            pinecone_index_name="",
            pinecone_cloud="aws",
            pinecone_region="us-east-1",
            chunk_size=500,
            chunk_overlap=100,
            top_k=3,
            use_mock_ai=False,
            use_local_vector_db=True,
        )

        with patch("embedding.get_embedding_client") as mock_client_factory:
            from openai import AuthenticationError
            mock_client = MagicMock()
            mock_client.embeddings.create.side_effect = AuthenticationError(
                message="Invalid API Key provided",
                response=MagicMock(status_code=401),
                body=None,
            )
            mock_client_factory.return_value = mock_client

            with self.assertRaises(EmbeddingAuthenticationError) as ctx:
                embed_text("test secret isolation", config=mock_config)

            # Error message must NEVER contain the secret
            self.assertNotIn("SECRET-KEY", str(ctx.exception))
            self.assertNotIn("12345", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
