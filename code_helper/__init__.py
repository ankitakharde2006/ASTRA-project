"""
code_helper — Top-level ADK Multi-Agent Assistant Package.

Exposes the root coordinator agent that manages the four specialized workers:
- onboarding_guide    : Project overview, 8 pipeline stages, safe env setup, run commands.
- architecture_mapper : Static AST codebase structure, directory listing, imports.
- code_searcher       : Semantic vector search & evidence retrieval via RAG.
- code_explainer      : In-depth file explanations using MCP Filesystem tools.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from coordinator import root_agent

__all__ = ["root_agent"]
