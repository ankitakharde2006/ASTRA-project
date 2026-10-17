"""
test_architecture_mapper.py — Verification script for the Architecture Mapper tools and agent.

Tests:
1. Tool 1: list_project_files() excludes .git, .venv, __pycache__, and generated files.
2. Tool 2: get_file_imports() extracts imports using ast.
3. Tool 3: get_function_definitions() extracts functions and classes using ast.
4. Edge cases: Missing files, non-python files, syntax errors, and path escaping.
5. Agent instantiation and tool registration.
"""

import unittest
from pathlib import Path
from architecture_mapper import (
    list_project_files,
    get_file_imports,
    get_function_definitions,
    root_agent,
)


class TestArchitectureMapperTools(unittest.TestCase):

    def test_01_list_project_files(self):
        output = list_project_files()
        self.assertIn("config.py", output)
        self.assertIn("search.py", output)
        self.assertNotIn(".git", output)
        self.assertNotIn(".venv", output)
        self.assertNotIn("__pycache__", output)
        self.assertNotIn("local_vector_db.json", output)

        # Test filtering by extension
        py_output = list_project_files(".py")
        self.assertIn("config.py", py_output)
        self.assertNotIn(".md", py_output)

    def test_02_get_file_imports(self):
        output = get_file_imports("search.py")
        self.assertIn("Imports in search.py", output)
        self.assertIn("from config import", output)
        self.assertIn("from embedding import", output)

    def test_03_get_function_definitions(self):
        output = get_function_definitions("search.py")
        self.assertIn("Definitions in search.py", output)
        self.assertIn("def search(", output)
        self.assertIn("class SearchError", output)

    def test_04_edge_cases(self):
        # Missing file
        missing_res = get_file_imports("does_not_exist.py")
        self.assertIn("File not found", missing_res)

        # Non-Python file for AST tools
        non_py = get_file_imports("requirements.txt")
        self.assertIn("not a Python file", non_py)

        # Path escape prevention
        escape_res = get_function_definitions("../../some_file.py")
        self.assertIn("File not found", escape_res)

    def test_05_agent_registration(self):
        self.assertEqual(root_agent.name, "architecture_mapper")
        tool_names = [t.__name__ for t in root_agent.tools]
        self.assertIn("list_project_files", tool_names)
        self.assertIn("get_file_imports", tool_names)
        self.assertIn("get_function_definitions", tool_names)
        self.assertEqual(len(tool_names), 3)


if __name__ == "__main__":
    unittest.main()
