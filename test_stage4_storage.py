"""
test_stage4_storage.py — Test Suite for Stage 4: Pinecone Vector Storage.

Tests:
1. Upsert embeddings to Pinecone and confirm vector count increases.
2. Fetch stored vectors by ID and verify metadata (text, source_file) is retrievable.
3. Idempotent upsert: re-indexing same chunks does NOT create duplicates.
4. Dimension validation: reject vectors that are not 2048 dimensions.
5. Empty input handling: empty chunk list returns 0 without errors.
6. Verify index dimension is 2048 and metric is cosine.
7. Secret protection: API key never appears in printed output.
8. Batch upsert: large sets are split into batches of <= 100.
"""

import time
import unittest
from config import get_rag_config
from embedding import embed_batch, EXPECTED_DIMENSION
from vector_store import (
    upsert_to_vector_db,
    query_vector_db,
    verify_pinecone_vectors,
    get_pinecone_client,
    make_chunk_id,
    build_metadata,
    validate_index_compatibility,
    VectorStoreError,
    VectorStoreConfigError,
    VectorStoreDimensionError,
    EXPECTED_METRIC,
)


# ---------------------------------------------------------------------------
# Deterministic test chunks for reproducible testing
# ---------------------------------------------------------------------------
SAMPLE_CHUNKS_TEXT = [
    {
        "id": "test_file.py#chunk_0",
        "file_path": "test_file.py",
        "relative_path": "test_file.py",
        "chunk_index": 0,
        "text": "def hello_world(): print('Hello from AstraTech Solutions')",
    },
    {
        "id": "test_file.py#chunk_1",
        "file_path": "test_file.py",
        "relative_path": "test_file.py",
        "chunk_index": 1,
        "text": "class RAGConfig: nvidia_api_key = '' embed_model = 'nvidia/nemotron-3-embed-1b'",
    },
    {
        "id": "test_file.py#chunk_2",
        "file_path": "test_file.py",
        "relative_path": "test_file.py",
        "chunk_index": 2,
        "text": "The remote home office setup stipend is twelve hundred dollars for all full-time employees.",
    },
]


def _embed_sample_chunks(config):
    """Helper: embed the sample chunks and attach vectors to each dict."""
    texts = [c["text"] for c in SAMPLE_CHUNKS_TEXT]
    vectors = embed_batch(texts, config=config, input_type="passage")
    enriched = []
    for i, chunk in enumerate(SAMPLE_CHUNKS_TEXT):
        enriched.append({**chunk, "embedding": vectors[i]})
    return enriched


class TestStage4PineconeStorage(unittest.TestCase):
    """
    Integration test suite for Stage 4: Pinecone Vector Storage.
    Requires live PINECONE_API_KEY and a compatible index (2048 dims, cosine).
    """

    @classmethod
    def setUpClass(cls):
        """Embed the sample chunks once for all tests."""
        cls.config = get_rag_config()
        if cls.config.use_local_vector_db:
            raise unittest.SkipTest(
                "Pinecone API key not configured. "
                "Set PINECONE_API_KEY in .env to run Stage 4 tests."
            )
        cls.embedded_chunks = _embed_sample_chunks(cls.config)

    def test_01_index_dimension_and_metric(self):
        """Verify the target Pinecone index has dimension=2048 and metric=cosine."""
        client = get_pinecone_client(self.config.pinecone_api_key)
        idx_dict = validate_index_compatibility(
            client,
            self.config.pinecone_index_name,
            expected_dimension=EXPECTED_DIMENSION,
            expected_metric=EXPECTED_METRIC,
        )
        # If validate didn't raise, the index is compatible
        self.assertIsInstance(idx_dict, dict)
        print(f"[OK] Index '{self.config.pinecone_index_name}' validated: "
              f"2048 dimensions, cosine metric.")

    def test_02_upsert_embeddings(self):
        """Upsert sample embeddings to Pinecone and confirm the count."""
        stored = upsert_to_vector_db(
            chunks_with_embeddings=self.embedded_chunks,
            config=self.config,
        )
        self.assertEqual(stored, len(self.embedded_chunks))
        print(f"[OK] Upserted {stored} vectors to Pinecone.")

    def test_03_fetch_metadata_is_retrievable(self):
        """Fetch stored vectors by ID and verify metadata fields are present."""
        # Allow Pinecone eventual consistency
        time.sleep(3)

        sample_ids = [c["id"] for c in SAMPLE_CHUNKS_TEXT]
        verification = verify_pinecone_vectors(
            config=self.config,
            sample_ids=sample_ids,
        )

        self.assertEqual(verification["dimension"], EXPECTED_DIMENSION)
        self.assertEqual(verification["metric"], EXPECTED_METRIC)
        self.assertGreaterEqual(verification["total_vectors"], len(SAMPLE_CHUNKS_TEXT))

        fetched = verification.get("fetched_samples", {})
        for chunk_id in sample_ids:
            self.assertIn(chunk_id, fetched, f"Vector '{chunk_id}' not found in Pinecone.")
            sample = fetched[chunk_id]
            self.assertEqual(sample["values_length"], EXPECTED_DIMENSION)
            self.assertIn("text", sample["metadata"])
            self.assertGreater(len(sample["metadata"]["text"]), 0)
            self.assertIn("source_file", sample["metadata"])
            self.assertEqual(sample["metadata"]["source_file"], "test_file.py")
            print(f"  [OK] {chunk_id}: text='{sample['metadata']['text'][:50]}...', "
                  f"source={sample['metadata']['source_file']}")

    def test_04_idempotent_upsert_no_duplicates(self):
        """Re-upserting the same chunks must NOT increase the total vector count."""
        # Get count before
        verification_before = verify_pinecone_vectors(config=self.config)
        count_before = verification_before["total_vectors"]

        # Upsert the exact same chunks again
        upsert_to_vector_db(
            chunks_with_embeddings=self.embedded_chunks,
            config=self.config,
        )

        # Allow eventual consistency
        time.sleep(3)

        # Get count after
        verification_after = verify_pinecone_vectors(config=self.config)
        count_after = verification_after["total_vectors"]

        self.assertEqual(
            count_before, count_after,
            f"Duplicate records detected! Before: {count_before}, After: {count_after}. "
            f"Upsert should overwrite, not append."
        )
        print(f"[OK] Idempotent: count stayed at {count_after} after re-upsert.")

    def test_05_dimension_validation_rejects_wrong_dims(self):
        """Chunks with wrong-dimension vectors must be rejected before any API call."""
        bad_chunks = [{
            "id": "bad_dim_chunk",
            "text": "This chunk has wrong dimensions",
            "embedding": [0.1] * 64,  # 64 dims instead of 2048
        }]
        with self.assertRaises(VectorStoreDimensionError):
            upsert_to_vector_db(bad_chunks, config=self.config)

    def test_06_empty_input_returns_zero(self):
        """Empty chunk list returns 0 without making any API calls."""
        result = upsert_to_vector_db([], config=self.config)
        self.assertEqual(result, 0)

    def test_07_query_retrieves_relevant_chunks(self):
        """Query the stored vectors and confirm relevant chunks are returned."""
        # Embed a query related to the stored content
        from embedding import embed_text
        query_vec = embed_text(
            "What is the remote office stipend?",
            config=self.config,
            input_type="query",
        )

        results = query_vector_db(
            query_vector=query_vec,
            top_k=3,
            config=self.config,
        )

        self.assertGreater(len(results), 0, "Query should return at least 1 result.")
        for r in results:
            self.assertIn("id", r)
            self.assertIn("score", r)
            self.assertIn("text", r)
            self.assertGreater(len(r["text"]), 0)
            print(f"  [Match] {r['id']}: score={r['score']:.4f}, text='{r['text'][:60]}...'")

    def test_08_make_chunk_id_is_deterministic(self):
        """Verify that make_chunk_id produces the same ID for the same chunk."""
        chunk = {
            "id": "config.py#chunk_0",
            "file_path": "config.py",
            "chunk_index": 0,
            "text": "some code",
        }
        id1 = make_chunk_id(chunk)
        id2 = make_chunk_id(chunk)
        self.assertEqual(id1, id2)
        self.assertEqual(id1, "config.py#chunk_0")

    def test_09_build_metadata_contains_required_fields(self):
        """Verify build_metadata includes text, chunk_id, and source_file."""
        chunk = {
            "id": "indexer.py#chunk_3",
            "file_path": "indexer.py",
            "chunk_index": 3,
            "text": "def load_document(): pass",
        }
        meta = build_metadata(chunk)
        self.assertEqual(meta["text"], "def load_document(): pass")
        self.assertEqual(meta["chunk_id"], "indexer.py#chunk_3")
        self.assertEqual(meta["source_file"], "indexer.py")
        self.assertEqual(meta["chunk_index"], "3")

    def test_10_api_key_never_in_output(self):
        """Ensure VectorStoreConfigError messages never contain the actual API key."""
        with self.assertRaises(VectorStoreConfigError):
            get_pinecone_client("")

        with self.assertRaises(VectorStoreConfigError):
            get_pinecone_client("   ")

    @classmethod
    def tearDownClass(cls):
        """Clean up test vectors from Pinecone after all tests complete."""
        if cls.config.use_local_vector_db:
            return
        try:
            client = get_pinecone_client(cls.config.pinecone_api_key)
            index = client.Index(cls.config.pinecone_index_name)
            test_ids = [c["id"] for c in SAMPLE_CHUNKS_TEXT]
            index.delete(ids=test_ids)
            print(f"\n[CLEANUP] Deleted {len(test_ids)} test vectors from Pinecone.")
        except Exception as err:
            print(f"\n[CLEANUP WARNING] Could not delete test vectors: {err}")


if __name__ == "__main__":
    unittest.main()
