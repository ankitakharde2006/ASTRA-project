"""
test_stage1.py — Test Suite for Stage 1: Reading Target Python Source Files.

Tests:
1. Reading relevant .py files from the project directory.
2. Proper exclusion of .venv, .git, __pycache__, and other generated directories.
3. Relative file path preservation alongside each file's content.
4. Safe handling of empty files (0-byte and whitespace).
5. Safe handling of unreadable files (decode/permission/IO errors).
6. Strict protection of .env (never read or exposed).
7. Preservation of existing load_document() behavior.
"""

import os
import shutil
import tempfile
import unittest
from pathlib import Path

from indexer import (
    load_python_files,
    read_python_files,
    load_source_files,
    load_document,
    is_excluded_directory,
    EXCLUDED_DIRECTORIES,
)


class TestStage1PythonFileIngestion(unittest.TestCase):
    """
    Validation test suite for Stage 1 of RAG: Reading target Python source files.
    """

    def setUp(self):
        # Create a temporary directory structure for testing edge cases
        self.temp_dir = tempfile.mkdtemp(prefix="test_stage1_")
        self.temp_path = Path(self.temp_dir)

    def tearDown(self):
        # Clean up temporary directory
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_load_current_project_python_files(self):
        """Test reading relevant .py files from the project directory."""
        current_dir = Path(__file__).resolve().parent
        files = load_python_files(current_dir)

        # Must return a non-empty list of file records
        self.assertIsInstance(files, list)
        self.assertGreater(len(files), 0)

        file_paths = [doc["file_path"] for doc in files]

        # Must include the core project files
        self.assertIn("config.py", file_paths)
        self.assertIn("indexer.py", file_paths)
        self.assertIn("main.py", file_paths)
        self.assertIn("query_pipeline.py", file_paths)
        self.assertIn("vector_store.py", file_paths)

        # None of the files should come from .venv, .git, or __pycache__
        for doc in files:
            p = doc["file_path"]
            self.assertTrue(p.endswith(".py"), f"Non-py file included: {p}")
            self.assertNotIn(".venv", p)
            self.assertNotIn(".git", p)
            self.assertNotIn("__pycache__", p)
            self.assertNotIn(".env", p)

            # Check required fields
            self.assertIn("file_path", doc)
            self.assertIn("relative_path", doc)
            self.assertIn("content", doc)
            self.assertGreater(len(doc["content"]), 0)

    def test_excluded_folders_isolated(self):
        """Test that .venv, .git, __pycache__, and generated folders are strictly excluded."""
        # Create mock file structure
        (self.temp_path / "valid_root.py").write_text("x = 1\n", encoding="utf-8")
        
        # Nested valid module
        src_dir = self.temp_path / "src" / "utils"
        src_dir.mkdir(parents=True)
        (src_dir / "helpers.py").write_text("def helper(): pass\n", encoding="utf-8")

        # Excluded directories
        venv_dir = self.temp_path / ".venv" / "Lib" / "site-packages"
        venv_dir.mkdir(parents=True)
        (venv_dir / "hidden_venv.py").write_text("bad = True\n", encoding="utf-8")

        git_dir = self.temp_path / ".git" / "hooks"
        git_dir.mkdir(parents=True)
        (git_dir / "hook.py").write_text("bad = True\n", encoding="utf-8")

        pycache_dir = self.temp_path / "src" / "__pycache__"
        pycache_dir.mkdir(parents=True)
        (pycache_dir / "module.cpython-313.py").write_text("bad = True\n", encoding="utf-8")

        build_dir = self.temp_path / "build"
        build_dir.mkdir(parents=True)
        (build_dir / "generated.py").write_text("bad = True\n", encoding="utf-8")

        dist_dir = self.temp_path / "dist"
        dist_dir.mkdir(parents=True)
        (dist_dir / "bundle.py").write_text("bad = True\n", encoding="utf-8")

        files = load_python_files(self.temp_path)
        loaded_names = [Path(doc["file_path"]).name for doc in files]

        self.assertIn("valid_root.py", loaded_names)
        self.assertIn("helpers.py", loaded_names)
        self.assertNotIn("hidden_venv.py", loaded_names)
        self.assertNotIn("hook.py", loaded_names)
        self.assertNotIn("module.cpython-313.py", loaded_names)
        self.assertNotIn("generated.py", loaded_names)
        self.assertNotIn("bundle.py", loaded_names)
        self.assertEqual(len(files), 2)

    def test_relative_path_preservation(self):
        """Test that relative paths are preserved alongside content."""
        sub_dir = self.temp_path / "pkg" / "submodule"
        sub_dir.mkdir(parents=True)
        test_content = "# Submodule code\nVERSION = '1.0.0'\n"
        (sub_dir / "module_a.py").write_text(test_content, encoding="utf-8")

        files = load_python_files(self.temp_path)
        self.assertEqual(len(files), 1)
        doc = files[0]

        expected_rel_path = str(Path("pkg") / "submodule" / "module_a.py")
        self.assertEqual(doc["file_path"], expected_rel_path)
        self.assertEqual(doc["relative_path"], expected_rel_path)
        self.assertEqual(doc["content"], test_content)

    def test_handle_empty_files_safely(self):
        """Test safe handling of 0-byte and whitespace-only files."""
        # Valid file
        (self.temp_path / "valid.py").write_text("a = 10\n", encoding="utf-8")
        # 0-byte empty file
        (self.temp_path / "empty.py").write_text("", encoding="utf-8")
        # Whitespace-only file
        (self.temp_path / "whitespace.py").write_text("   \n\t  \n", encoding="utf-8")

        # Must not crash, should safely skip empty files by default
        files = load_python_files(self.temp_path)
        self.assertEqual(len(files), 1)
        self.assertEqual(files[0]["file_path"], "valid.py")

        # If include_empty=True is requested, it safely includes them without error
        files_with_empty = load_python_files(self.temp_path, include_empty=True)
        self.assertEqual(len(files_with_empty), 3)

    def test_handle_unreadable_files_safely(self):
        """Test safe handling when encountering unreadable / corrupted files."""
        (self.temp_path / "valid.py").write_text("print('hello')", encoding="utf-8")

        # Write arbitrary invalid UTF-8 bytes to simulate binary or corrupted files
        unreadable_file = self.temp_path / "corrupted.py"
        with open(unreadable_file, "wb") as f:
            f.write(b"\xff\xfe\x00\x00\x80\x81\x82\x83\x99\xaa\xbb\xcc\xdd\xee")

        # Must not raise an exception
        try:
            files = load_python_files(self.temp_path)
            loaded_paths = [doc["file_path"] for doc in files]
            self.assertIn("valid.py", loaded_paths)
        except Exception as e:
            self.fail(f"load_python_files crashed on unreadable file: {e}")

    def test_env_files_and_secrets_protected(self):
        """Test that .env files and variables are never read or returned."""
        secret_content = "SUPER_SECRET_API_KEY=nvapi-secret-123456789\nPINECONE_KEY=pc-secret"
        (self.temp_path / ".env").write_text(secret_content, encoding="utf-8")
        (self.temp_path / ".env.local").write_text(secret_content, encoding="utf-8")
        (self.temp_path / "valid.py").write_text("x = 42\n", encoding="utf-8")

        files = load_python_files(self.temp_path)
        loaded_paths = [doc["file_path"] for doc in files]

        self.assertNotIn(".env", loaded_paths)
        self.assertNotIn(".env.local", loaded_paths)
        for doc in files:
            self.assertNotIn("SUPER_SECRET_API_KEY", doc["content"])

    def test_existing_load_document_preserved(self):
        """Ensure load_document() still works exactly as before for backwards compatibility."""
        handbook_path = Path(__file__).resolve().parent / "data" / "astratech_employee_handbook.txt"
        if handbook_path.exists():
            text = load_document(str(handbook_path))
            self.assertIsInstance(text, str)
            self.assertIn("ASTRATECH SOLUTIONS", text)

    def test_aliases_available(self):
        """Ensure aliases read_python_files and load_source_files work."""
        self.assertIs(read_python_files, load_python_files)
        self.assertIs(load_source_files, load_python_files)


if __name__ == "__main__":
    unittest.main()
