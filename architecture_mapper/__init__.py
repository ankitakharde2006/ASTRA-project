"""
architecture_mapper — Google ADK agent that maps the structure of the
Naive RAG project using pure-Python static analysis tools.

Tools (no AI calls — deterministic Python only):
    1. list_project_files   — walk the project tree, skip noise directories.
    2. get_file_imports     — extract import / from-import statements with ast.
    3. get_function_definitions — list functions, classes, and their line numbers.

Architecture:
    The agent receives a user question, decides which tool(s) to call, and
    then synthesises an answer from the structured tool output.  All heavy
    lifting happens in plain Python; the LLM only interprets results.
"""

from __future__ import annotations

import ast
import os
import sys
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Project root — anchored to the Naive_RAG directory that contains this agent
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Directories to always skip when traversing the project tree
_SKIP_DIRS = {".git", ".venv", "__pycache__", ".adk", "node_modules", ".tox"}

# File extensions considered "relevant source files"
_SOURCE_EXTENSIONS = {".py", ".md", ".txt", ".json", ".yaml", ".yml", ".toml",
                      ".cfg", ".ini", ".ipynb"}


# ═══════════════════════════════════════════════════════════════════════════
# Tool 1 — list_project_files
# ═══════════════════════════════════════════════════════════════════════════
def list_project_files(extension_filter: Optional[str] = None) -> str:
    """List relevant source files in the Naive RAG project.

    Walks the project tree starting from the project root, skipping
    noise directories (.git, .venv, __pycache__, etc.) and generated
    data files (local_vector_db.json).

    Args:
        extension_filter: Optional file extension to filter by,
            e.g. ".py" to list only Python files.  When omitted all
            recognised source extensions are included.

    Returns:
        A newline-separated list of relative file paths from the project
        root, or an error / "no files found" message.
    """
    try:
        found: list[str] = []
        for dirpath, dirnames, filenames in os.walk(_PROJECT_ROOT):
            # Prune noise directories in-place so os.walk skips them
            dirnames[:] = [
                d for d in dirnames if d not in _SKIP_DIRS
            ]

            for fname in sorted(filenames):
                # Skip known generated / binary data files
                if fname in {"local_vector_db.json"}:
                    continue

                fpath = Path(dirpath) / fname
                ext = fpath.suffix.lower()

                # Apply extension filter if provided
                if extension_filter:
                    normalised = extension_filter if extension_filter.startswith(".") else f".{extension_filter}"
                    if ext != normalised.lower():
                        continue
                elif ext not in _SOURCE_EXTENSIONS:
                    continue

                rel = fpath.relative_to(_PROJECT_ROOT)
                found.append(str(rel))

        if not found:
            qualifier = f" with extension '{extension_filter}'" if extension_filter else ""
            return f"No source files found{qualifier} in the project."

        header = f"Found {len(found)} file(s) in {_PROJECT_ROOT.name}:\n"
        return header + "\n".join(f"  {f}" for f in found)

    except Exception as exc:
        return f"Error listing project files: {exc}"


# ═══════════════════════════════════════════════════════════════════════════
# Tool 2 — get_file_imports
# ═══════════════════════════════════════════════════════════════════════════
def get_file_imports(file_path: str) -> str:
    """Extract all import statements from a Python source file.

    Parses the file with Python's ``ast`` module and returns every
    ``import`` and ``from … import`` statement with the module names
    and imported symbols.

    Args:
        file_path: Path to the Python file, either absolute or relative
            to the project root.

    Returns:
        A structured, human-readable list of imports, or an error
        message if the file is missing or contains syntax errors.
    """
    resolved = _resolve_file_path(file_path)
    if resolved is None:
        return f"File not found: '{file_path}' (must be inside the project root)."

    if resolved.suffix.lower() != ".py":
        return f"'{file_path}' is not a Python file — only .py files can be parsed for imports."

    try:
        source = resolved.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return f"Could not read '{file_path}': {exc}"

    try:
        tree = ast.parse(source, filename=str(resolved))
    except SyntaxError as exc:
        return f"Syntax error in '{file_path}' (line {exc.lineno}): {exc.msg}"

    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                as_part = f" as {alias.asname}" if alias.asname else ""
                imports.append(f"  line {node.lineno}: import {alias.name}{as_part}")

        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            level_dots = "." * (node.level or 0)
            names = ", ".join(
                f"{a.name}" + (f" as {a.asname}" if a.asname else "")
                for a in node.names
            )
            imports.append(f"  line {node.lineno}: from {level_dots}{module} import {names}")

    if not imports:
        return f"No import statements found in '{file_path}'."

    rel = resolved.relative_to(_PROJECT_ROOT)
    header = f"Imports in {rel} ({len(imports)} statement(s)):\n"
    return header + "\n".join(imports)


# ═══════════════════════════════════════════════════════════════════════════
# Tool 3 — get_function_definitions
# ═══════════════════════════════════════════════════════════════════════════
def get_function_definitions(file_path: str) -> str:
    """List all function and class definitions in a Python source file.

    Uses Python's ``ast`` module to extract every ``def``, ``async def``,
    and ``class`` statement, along with its line number range,
    decorators, and (for functions) the argument list.

    Args:
        file_path: Path to the Python file, either absolute or relative
            to the project root.

    Returns:
        A structured, human-readable listing of all definitions, or an
        error message if the file is missing or has syntax errors.
    """
    resolved = _resolve_file_path(file_path)
    if resolved is None:
        return f"File not found: '{file_path}' (must be inside the project root)."

    if resolved.suffix.lower() != ".py":
        return f"'{file_path}' is not a Python file — only .py files can be parsed."

    try:
        source = resolved.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return f"Could not read '{file_path}': {exc}"

    try:
        tree = ast.parse(source, filename=str(resolved))
    except SyntaxError as exc:
        return f"Syntax error in '{file_path}' (line {exc.lineno}): {exc.msg}"

    definitions: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            kind = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
            args_str = _format_args(node.args)
            decorators = _format_decorators(node)
            end_line = getattr(node, "end_lineno", "?")
            definitions.append(
                f"  {kind} {node.name}({args_str})  "
                f"[lines {node.lineno}–{end_line}]"
                + (f"  decorators: {decorators}" if decorators else "")
            )

        elif isinstance(node, ast.ClassDef):
            bases = ", ".join(_unparse_node(b) for b in node.bases) if node.bases else ""
            decorators = _format_decorators(node)
            end_line = getattr(node, "end_lineno", "?")
            base_part = f"({bases})" if bases else ""
            definitions.append(
                f"  class {node.name}{base_part}  "
                f"[lines {node.lineno}–{end_line}]"
                + (f"  decorators: {decorators}" if decorators else "")
            )

    if not definitions:
        return f"No function or class definitions found in '{file_path}'."

    rel = resolved.relative_to(_PROJECT_ROOT)
    header = f"Definitions in {rel} ({len(definitions)} item(s)):\n"
    return header + "\n".join(definitions)


# ═══════════════════════════════════════════════════════════════════════════
# Internal helpers (not exposed as tools)
# ═══════════════════════════════════════════════════════════════════════════
def _resolve_file_path(file_path: str) -> Optional[Path]:
    """Resolve a user-supplied path to an absolute path inside the project.

    Accepts both absolute paths and paths relative to the project root.
    Returns ``None`` if the file doesn't exist or escapes the project root.
    """
    p = Path(file_path)

    if not p.is_absolute():
        p = _PROJECT_ROOT / p

    p = p.resolve()

    # Security: ensure path doesn't escape the project root
    try:
        p.relative_to(_PROJECT_ROOT)
    except ValueError:
        return None

    if not p.is_file():
        return None

    return p


def _format_args(args: ast.arguments) -> str:
    """Format a function's argument list for display."""
    parts: list[str] = []
    for arg in args.args:
        parts.append(arg.arg)
    if args.vararg:
        parts.append(f"*{args.vararg.arg}")
    for arg in args.kwonlyargs:
        parts.append(arg.arg)
    if args.kwarg:
        parts.append(f"**{args.kwarg.arg}")
    return ", ".join(parts)


def _format_decorators(node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> str:
    """Return a comma-separated string of decorator names."""
    names: list[str] = []
    for dec in node.decorator_list:
        names.append(_unparse_node(dec))
    return ", ".join(names)


def _unparse_node(node: ast.expr) -> str:
    """Best-effort conversion of an AST node back to source text."""
    try:
        return ast.unparse(node)
    except Exception:
        return "?"


# ═══════════════════════════════════════════════════════════════════════════
# ADK Agent wiring
# ═══════════════════════════════════════════════════════════════════════════
# Reuse the same shared config & LiteLLM model pattern as code_searcher
_NAIVE_RAG_DIR = str(Path(__file__).resolve().parent.parent)
if _NAIVE_RAG_DIR not in sys.path:
    sys.path.insert(0, _NAIVE_RAG_DIR)

from config import get_rag_config  # noqa: E402
from google.adk import Agent  # noqa: E402
from google.adk.models import LiteLlm  # noqa: E402

_rag_config = get_rag_config()

class SingleToolCallLiteLlm(LiteLlm):
    """LiteLlm adapter that enforces single tool-calls for NVIDIA NIM models."""

    async def generate_content_async(self, llm_request, stream=False):
        async for response in super().generate_content_async(llm_request, stream=stream):
            if response.content and response.content.parts:
                fn_calls = [p for p in response.content.parts if p.function_call is not None]
                if len(fn_calls) > 1:
                    first_fn = False
                    filtered = []
                    for p in response.content.parts:
                        if p.function_call is not None:
                            if not first_fn:
                                filtered.append(p)
                                first_fn = True
                        else:
                            filtered.append(p)
                    response.content.parts = filtered
            yield response


_litellm_model = SingleToolCallLiteLlm(
    model=f"openai/{_rag_config.chat_model}",
    api_key=_rag_config.nvidia_api_key,
    api_base=_rag_config.nvidia_base_url,
    temperature=0.1,
    max_tokens=2048,
    parallel_tool_calls=False,
)

_INSTRUCTION = """\
You are **Architecture Mapper**, a technical assistant that analyses the
structure of the Naive RAG Python project.

## Available tools
| Tool | Purpose |
|---|---|
| **list_project_files** | List source files in the project, optionally filtered by extension |
| **get_file_imports** | Extract import statements from a Python file |
| **get_function_definitions** | List functions, classes, and their line numbers from a Python file |

## Workflow
1. Use **list_project_files** first to discover what files exist.
2. Then use **get_file_imports** and/or **get_function_definitions** on
   specific files the user asks about.
3. Synthesise your answer **strictly from the tool output**.

## Rules
- **CRITICAL**: Call only ONE tool at a time. Never invoke multiple tools in parallel or simultaneously.
- **Never invent** file names, function names, class names, import paths,
  or line numbers that do not appear in the tool output.
- **Never guess** what a file contains — always call the appropriate tool.
- If a tool returns an error, report the error clearly to the user.
- Keep answers concise, structured, and factual.
- Do not call the same tool with the same arguments more than once per question.
"""

root_agent = Agent(
    name="architecture_mapper",
    description=(
        "Maps the architecture of the Naive RAG project by listing files, "
        "extracting imports, and discovering function/class definitions "
        "using pure-Python static analysis."
    ),
    model=_litellm_model,
    instruction=_INSTRUCTION,
    tools=[list_project_files, get_file_imports, get_function_definitions],
)
