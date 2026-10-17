"""
test_stage2_chunking.py — Focused Unit Test Suite for Stage 2: Chunking.

Covers:
1. Short files (length < chunk_size) -> Exactly 1 chunk, index 0, full content.
2. Long files (length >> chunk_size) -> Sequential chunk_index, file_path preserved.
3. Empty files (0-byte and whitespace) -> 0 chunks, no empty chunks produced.
4. Overlapping chunks -> Validates exact character overlap between consecutive chunks.
5. Infinite loop prevention -> Rejects chunk_overlap >= chunk_size and non-positive sizes.
6. Integration with Stage 1 file reader -> load_python_files() to chunk_files().
7. Context preservation verification -> Ensures code context spanning boundaries is preserved.
"""

import os
import shutil
import tempfile
import unittest
from pathlib import Path

from indexer import (
    chunk_document,
    chunk_files,
    chunk_python_files,
    load_python_files,
)


class TestStage2Chunking(unittest.TestCase):
    """
    Focused test cases for custom RAG chunking stage.
    """

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="test_stage2_")
        self.temp_path = Path(self.temp_dir)

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_short_file(self):
        """Short file (less than chunk_size) must produce exactly 1 chunk with index 0."""
        short_code = (
            "def add(a: int, b: int) -> int:\n"
            "    \"\"\"Add two numbers.\"\"\"\n"
            "    return a + b\n"
        )
        file_path = "math_utils.py"
        chunks = chunk_document(
            text=short_code,
            chunk_size=500,
            chunk_overlap=100,
            file_path=file_path,
        )

        self.assertEqual(len(chunks), 1, "Short file should produce exactly 1 chunk.")
        chunk = chunks[0]
        self.assertEqual(chunk["chunk_index"], 0)
        self.assertEqual(chunk["file_path"], file_path)
        self.assertEqual(chunk["text"], short_code.strip())
        self.assertIn("id", chunk)
        self.assertEqual(chunk["id"], f"{file_path}#chunk_0")

    def test_long_file(self):
        """Long file must produce multiple chunks with sequential indices and preserved file path."""
        # Generate a simulated long python module (~1,800 characters)
        lines = [
            f"def function_{i}(x: int) -> int:\n"
            f"    \"\"\"Docstring for function {i} with calculation details.\"\"\"\n"
            f"    result = x * {i} + 42\n"
            f"    return result\n"
            for i in range(25)
        ]
        long_code = "\n".join(lines)
        file_path = "services/processor.py"

        chunk_size = 350
        chunk_overlap = 75
        chunks = chunk_document(
            text=long_code,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            file_path=file_path,
        )

        self.assertGreater(len(chunks), 1, "Long file should produce multiple chunks.")

        # Check sequential indexing and file path retention
        for expected_idx, chunk in enumerate(chunks):
            self.assertEqual(chunk["chunk_index"], expected_idx)
            self.assertEqual(chunk["file_path"], file_path)
            self.assertEqual(chunk["id"], f"{file_path}#chunk_{expected_idx}")
            self.assertGreater(len(chunk["text"]), 0, "No chunk should be empty.")
            self.assertLessEqual(len(chunk["text"]), chunk_size + 10)

    def test_empty_and_whitespace_files(self):
        """Empty files (0 bytes) and whitespace-only files must produce 0 chunks."""
        # Completely empty text
        empty_chunks = chunk_document(
            text="",
            chunk_size=350,
            chunk_overlap=50,
            file_path="empty.py",
        )
        self.assertEqual(empty_chunks, [], "Empty text should yield 0 chunks.")

        # Whitespace-only text (spaces, tabs, newlines)
        whitespace_chunks = chunk_document(
            text="   \n\n\t\t   \n   ",
            chunk_size=350,
            chunk_overlap=50,
            file_path="whitespace.py",
        )
        self.assertEqual(whitespace_chunks, [], "Whitespace text should yield 0 chunks.")

    def test_overlapping_chunks_preserves_boundary_context(self):
        """Verify that consecutive chunks share exact overlapping characters."""
        # Create a document where a critical function header lands on a chunk boundary
        raw_text = (
            "ALPHA_TOKEN = 1000\n"
            "BETA_TOKEN = 2000\n"
            "GAMMA_TOKEN = 3000\n"
            "def calculate_total_tax(amount: float, rate: float) -> float:\n"
            "    # Critical calculation logic starts here\n"
            "    tax = amount * rate\n"
            "    return tax\n"
            "DELTA_TOKEN = 4000\n"
            "EPSILON_TOKEN = 5000\n"
        )
        chunk_size = 100
        chunk_overlap = 35
        step = chunk_size - chunk_overlap  # 65

        chunks = chunk_document(
            text=raw_text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            file_path="tax.py",
        )
        self.assertGreater(len(chunks), 1)

        # For every adjacent pair of chunks, verify the overlapping substring
        for i in range(len(chunks) - 1):
            curr_chunk = chunks[i]
            next_chunk = chunks[i + 1]

            # In the raw text, the overlap starts at next_chunk['start_char']
            # and ends at curr_chunk['end_char']
            overlap_start = next_chunk["start_char"]
            overlap_end = curr_chunk["end_char"]
            self.assertGreater(overlap_end, overlap_start, "Expected positive overlap window.")

            overlap_text = raw_text[overlap_start:overlap_end]
            self.assertIn(
                overlap_text.strip(),
                curr_chunk["text"],
                "Tail of current chunk must contain the overlapping boundary text.",
            )
            self.assertIn(
                overlap_text.strip(),
                next_chunk["text"],
                "Head of next chunk must contain the overlapping boundary text.",
            )

    def test_infinite_loop_prevention(self):
        """Ensure invalid chunk_size and chunk_overlap combinations are strictly rejected."""
        # Overlap equal to chunk size (step = 0 -> infinite loop)
        with self.assertRaises(ValueError):
            chunk_document("some test code", chunk_size=100, chunk_overlap=100)

        # Overlap greater than chunk size (step < 0 -> backwards traversal)
        with self.assertRaises(ValueError):
            chunk_document("some test code", chunk_size=100, chunk_overlap=150)

        # Zero or negative chunk size
        with self.assertRaises(ValueError):
            chunk_document("some test code", chunk_size=0, chunk_overlap=0)

        with self.assertRaises(ValueError):
            chunk_document("some test code", chunk_size=-50, chunk_overlap=10)

        # Negative overlap
        with self.assertRaises(ValueError):
            chunk_document("some test code", chunk_size=100, chunk_overlap=-10)

    def test_chunk_files_with_reader_integration(self):
        """Test end-to-end integration: read files via load_python_files then chunk them."""
        # 1. Create a short file
        (self.temp_path / "mod_short.py").write_text("X = 1\nY = 2\n", encoding="utf-8")

        # 2. Create a long file
        long_content = "\n".join([f"var_{i} = 'constant_value_{i}'" for i in range(50)])
        (self.temp_path / "mod_long.py").write_text(long_content, encoding="utf-8")

        # 3. Create an empty file
        (self.temp_path / "mod_empty.py").write_text("", encoding="utf-8")

        # Read files using Stage 1 reader
        files = load_python_files(self.temp_path)
        self.assertEqual(len(files), 2, "Stage 1 reader correctly skips empty file by default.")

        # Chunk files using Stage 2 chunker
        chunks = chunk_files(files, chunk_size=200, chunk_overlap=40)
        self.assertGreater(len(chunks), 1)

        # Verify all chunks have source file_path and chunk_index
        short_chunks = [c for c in chunks if c["file_path"] == "mod_short.py"]
        long_chunks = [c for c in chunks if c["file_path"] == "mod_long.py"]

        self.assertEqual(len(short_chunks), 1)
        self.assertEqual(short_chunks[0]["chunk_index"], 0)

        self.assertGreater(len(long_chunks), 1)
        for idx, c in enumerate(long_chunks):
            self.assertEqual(c["chunk_index"], idx)
            self.assertEqual(c["file_path"], "mod_long.py")

    def test_alias_chunk_python_files(self):
        """Verify chunk_python_files alias works identically."""
        self.assertIs(chunk_python_files, chunk_files)


if __name__ == "__main__":
    unittest.main()
