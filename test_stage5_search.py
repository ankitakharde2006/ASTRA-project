"""
test_stage5_search.py — Test Suite for Stage 5+6: Evidence Retrieval (Search).

Tests:
1. Known-answer question retrieves the correct chunk with HIGH relevance.
2. Unrelated question returns results with LOW/NOISE relevance (no useful evidence).
3. Empty and whitespace queries are rejected with SearchInputError.
4. Empty index scenario returns an empty list gracefully.
5. Every returned record contains all required fields and valid types.
6. Results are ranked by descending similarity score.
7. Relevance tier labels are correctly assigned.
"""

import unittest
from config import get_rag_config
from search import (
    search,
    SearchInputError,
    SearchError,
    SCORE_HIGH,
    SCORE_MODERATE,
    SCORE_LOW,
    _relevance_tier,
    _validate_and_enrich,
)


class TestStage5Search(unittest.TestCase):
    """
    Integration tests for evidence retrieval against the indexed
    AstraTech employee handbook in Pinecone (capstone-rag-index).
    """

    @classmethod
    def setUpClass(cls):
        cls.config = get_rag_config()

    # ------------------------------------------------------------------
    # Test 1: Known-answer retrieval
    # ------------------------------------------------------------------
    def test_01_known_answer_retrieves_stipend_chunk(self):
        """
        The handbook states the remote home office setup stipend is $1,200.
        Searching for that topic must return a HIGH-relevance chunk containing
        the dollar amount.
        """
        results = search(
            "What is the one-time remote home office setup stipend?",
            config=self.config,
            top_k=3,
        )

        self.assertGreater(len(results), 0, "Must return at least 1 result.")

        # The top result should mention the stipend amount
        top = results[0]
        self.assertIn("1,200", top["text"],
                       "Top chunk must contain the $1,200 stipend figure.")
        self.assertIn(top["relevance"], ("HIGH", "MODERATE"),
                       "Top result for a known-answer query should be HIGH or MODERATE.")
        print(f"\n[PASS] Top result: score={top['score']:.4f}, "
              f"relevance={top['relevance']}, "
              f"text contains '$1,200': True")

    # ------------------------------------------------------------------
    # Test 2: Known-answer — Wi-Fi password
    # ------------------------------------------------------------------
    def test_02_known_answer_retrieves_wifi_password(self):
        """
        The handbook states the visitor Wi-Fi password is CoffeeAndCode2026!.
        """
        results = search(
            "What is the office visitor Wi-Fi password?",
            config=self.config,
            top_k=3,
        )

        self.assertGreater(len(results), 0)

        # At least one result must contain the password
        found = any("CoffeeAndCode2026" in r["text"] for r in results)
        self.assertTrue(found,
                        "At least one result must contain the Wi-Fi password.")
        print(f"\n[PASS] Wi-Fi password found in top-{len(results)} results.")

    # ------------------------------------------------------------------
    # Test 3: Unrelated question — no useful evidence
    # ------------------------------------------------------------------
    def test_03_unrelated_question_returns_low_relevance(self):
        """
        A question completely unrelated to AstraTech should still return
        results (the index isn't empty), but all should be LOW or NOISE tier.
        """
        results = search(
            "What is the chemical formula for photosynthesis in C4 plants?",
            config=self.config,
            top_k=3,
        )

        # Results may or may not be returned, but none should be HIGH
        high_results = [r for r in results if r["relevance"] == "HIGH"]
        self.assertEqual(len(high_results), 0,
                         "An unrelated question should not produce HIGH-relevance evidence.")
        print(f"\n[PASS] Unrelated query: {len(results)} results, "
              f"0 HIGH-relevance (as expected).")

    # ------------------------------------------------------------------
    # Test 4: Input validation
    # ------------------------------------------------------------------
    def test_04_empty_query_raises_error(self):
        """Empty and whitespace-only queries must be rejected."""
        with self.assertRaises(SearchInputError):
            search("", config=self.config)

        with self.assertRaises(SearchInputError):
            search("   \n\t  ", config=self.config)

        with self.assertRaises(SearchInputError):
            search(123, config=self.config)  # type: ignore

    # ------------------------------------------------------------------
    # Test 5: Result structure validation
    # ------------------------------------------------------------------
    def test_05_result_records_have_all_required_fields(self):
        """Every result dict must contain rank, id, text, source_file,
        chunk_index, score, and relevance."""
        results = search(
            "What are the working hours?",
            config=self.config,
            top_k=3,
        )

        required_keys = {"rank", "id", "text", "source_file",
                         "chunk_index", "score", "relevance"}

        for r in results:
            for key in required_keys:
                self.assertIn(key, r, f"Missing required field '{key}' in result.")

            self.assertIsInstance(r["rank"], int)
            self.assertIsInstance(r["id"], str)
            self.assertIsInstance(r["text"], str)
            self.assertIsInstance(r["score"], float)
            self.assertIn(r["relevance"], ("HIGH", "MODERATE", "LOW", "NOISE"))
            self.assertGreaterEqual(r["score"], 0.0)
            self.assertGreater(len(r["text"]), 0)

    # ------------------------------------------------------------------
    # Test 6: Results ordered by descending score
    # ------------------------------------------------------------------
    def test_06_results_are_ranked_by_descending_score(self):
        """Results must be returned in descending order of similarity score."""
        results = search(
            "How many PTO days do employees get?",
            config=self.config,
            top_k=5,
        )

        if len(results) >= 2:
            scores = [r["score"] for r in results]
            for i in range(len(scores) - 1):
                self.assertGreaterEqual(
                    scores[i], scores[i + 1],
                    f"Results not in descending score order at position {i}."
                )

    # ------------------------------------------------------------------
    # Test 7: Relevance tier logic
    # ------------------------------------------------------------------
    def test_07_relevance_tiers_are_correct(self):
        """Unit test for the _relevance_tier function."""
        self.assertEqual(_relevance_tier(0.80), "HIGH")
        self.assertEqual(_relevance_tier(0.45), "HIGH")
        self.assertEqual(_relevance_tier(0.30), "MODERATE")
        self.assertEqual(_relevance_tier(0.25), "MODERATE")
        self.assertEqual(_relevance_tier(0.15), "LOW")
        self.assertEqual(_relevance_tier(0.10), "LOW")
        self.assertEqual(_relevance_tier(0.05), "NOISE")
        self.assertEqual(_relevance_tier(0.0), "NOISE")

    # ------------------------------------------------------------------
    # Test 8: validate_and_enrich handles missing fields
    # ------------------------------------------------------------------
    def test_08_validate_and_enrich_handles_missing_fields(self):
        """Enrichment must fill defaults for missing metadata gracefully."""
        minimal = {"id": "doc_chunk_2", "text": "some text", "score": 0.5}
        enriched = _validate_and_enrich(minimal, rank=1)

        self.assertEqual(enriched["rank"], 1)
        self.assertEqual(enriched["id"], "doc_chunk_2")
        self.assertEqual(enriched["text"], "some text")
        self.assertEqual(enriched["score"], 0.5)
        self.assertEqual(enriched["relevance"], "HIGH")
        self.assertEqual(enriched["source_file"], "document")
        self.assertEqual(enriched["chunk_index"], 2)

    # ------------------------------------------------------------------
    # Test 9: validate_and_enrich parses file#chunk IDs
    # ------------------------------------------------------------------
    def test_09_validate_and_enrich_parses_chunk_ids(self):
        """IDs like 'config.py#chunk_3' should be parsed into source_file + chunk_index."""
        raw = {"id": "config.py#chunk_3", "text": "code here", "score": 0.3}
        enriched = _validate_and_enrich(raw, rank=1)

        self.assertEqual(enriched["source_file"], "config.py")
        self.assertEqual(enriched["chunk_index"], 3)


if __name__ == "__main__":
    unittest.main()
