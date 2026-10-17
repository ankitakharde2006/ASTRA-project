"""
onboarding_guide — Google ADK agent that onboards new developers to the
Naive RAG project.

Its purpose:
    - Explain what the project does (8-stage Naive RAG pipeline).
    - Recommend important files to read first.
    - Guide safe environment configuration (without exposing secrets).
    - Provide verified commands to index, run, and test the project.
    - Suggest the best first steps when learning the codebase.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Make parent Naive_RAG directory importable
# ---------------------------------------------------------------------------
_NAIVE_RAG_DIR = str(Path(__file__).resolve().parent.parent)
if _NAIVE_RAG_DIR not in sys.path:
    sys.path.insert(0, _NAIVE_RAG_DIR)

from config import get_rag_config  # noqa: E402
from google.adk import Agent  # noqa: E402
from google.adk.models import LiteLlm  # noqa: E402

_rag_config = get_rag_config()


# ═══════════════════════════════════════════════════════════════════════════
# Onboarding Tools
# ═══════════════════════════════════════════════════════════════════════════
def get_project_summary() -> str:
    """Return the executive summary of the AstraTech Naive RAG project.

    Includes the 8-stage architecture pipeline (offline indexing + online querying)
    and key files to read first.

    Returns:
        Structured project summary text.
    """
    return """
=== PROJECT OVERVIEW ===
AstraTech Naive RAG Pipeline — An educational 8-stage Retrieval-Augmented
Generation system using NVIDIA NIM (Embeddings & Generation) and Pinecone
(Vector Database), with built-in offline simulation mode.

=== 8 ARCHITECTURE PIPELINE STAGES ===
[Offline Indexing Pipeline — indexer.py]
  Stage 1: Document Ingestion  -> load_document() / load_python_files()
  Stage 2: Chunking            -> chunk_document() / chunk_files() (Sliding window)
  Stage 3: Embedding           -> embed_chunks() (NVIDIA NIM 2048-dim vectors)
  Stage 4: Storage             -> store_in_vector_db() (Pinecone / local JSON fallback)

[Online Query Pipeline — query_pipeline.py]
  Stage 5: Query Embedding     -> embed_query() / search.search()
  Stage 6: Retrieval           -> retrieve_top_k() (Cosine similarity ranking)
  Stage 7: Context Injection   -> build_prompt_with_context() (Augmented prompt)
  Stage 8: Grounded Generation -> generate_answer() (NVIDIA NIM LLM synthesis)

=== KEY FILES TO READ FIRST ===
1. Day4_Naive_RAG_Project_Build_Spec.md : Architecture specification and 8-box map
2. config.py                            : Configuration loader and offline fallback modes
3. indexer.py                           : Offline ingestion, chunking, embedding pipeline
4. search.py                            : Core retrieval stage (similarity scores and tiers)
5. query_pipeline.py                    : Live query augmentation and answer generation
6. main.py                              : Interactive demo runner and comparison scorecard
""".strip()


def get_setup_and_run_instructions(topic: Optional[str] = None) -> str:
    """Return verified developer instructions for environment setup, indexing, running, and learning.

    Args:
        topic: Specific topic to filter by: 'env', 'index', 'run', or 'learning'.
               If omitted or None, returns all topics.

    Returns:
        Formatted instructions with exact verified CLI commands.
    """
    env_text = """
[Topic: Environment Setup (env)]
1. Create a `.env` file in the Naive_RAG directory with:
     NVIDIA_API_KEY=<your-nvidia-nim-key>
     PINECONE_API_KEY=<your-pinecone-key>
     PINECONE_INDEX_NAME=capstone-rag-index
     PINECONE_CLOUD=aws
     PINECONE_REGION=us-east-1
2. SAFETY: Never commit .env or paste real API keys into code or chat.
3. OFFLINE MODE: If keys are omitted, config.py automatically falls back to
   safe offline simulation (use_mock_ai=True, local JSON vector store).
""".strip()

    index_text = """
[Topic: Indexing Commands (index)]
  python indexer.py           # Full document indexing (Employee Handbook)
  python main.py --index      # Index via main CLI
  python indexer.py --stage1  # Test document loading only
  python indexer.py --stage2  # Test chunking only
  python indexer.py --stage3  # Test embedding via NVIDIA NIM
""".strip()

    run_text = """
[Topic: Running Instructions (run)]
  python main.py              # Full interactive live demo
  python main.py --compare    # Side-by-side: Raw LLM vs Naive RAG
  python main.py --query "What is the remote home office setup stipend?"
  python main.py --mock       # Run in offline mock mode (no API calls)
""".strip()

    learning_text = """
[Topic: Learning Path (learning)]
  Step A: Run test suites to verify local environment:
    python test_stage1.py
    python test_stage5_search.py
    python test_architecture_mapper.py
  Step B: Run side-by-side comparison:
    python main.py --compare
  Step C: Test the ADK workers:
    adk run code_searcher
    adk run architecture_mapper
    adk run onboarding_guide
""".strip()

    normalized_topic = (topic or "").strip().lower()

    if normalized_topic == "env":
        return env_text
    elif normalized_topic == "index":
        return index_text
    elif normalized_topic == "run":
        return run_text
    elif normalized_topic == "learning":
        return learning_text
    else:
        return f"{env_text}\n\n{index_text}\n\n{run_text}\n\n{learning_text}"


def list_project_files() -> str:
    """List the primary project files and directories in Naive RAG.

    Returns:
        Structured list of core project files.
    """
    return """
Core Project Files:
- config.py              : Centralized environment & hyperparameter configuration
- indexer.py             : Offline stages 1-4 (Ingestion, Chunking, Embedding, Storage)
- embedding.py           : NVIDIA NIM 2048-dim vector embedding caller
- vector_store.py        : Pinecone serverless vector index manager with offline JSON fallback
- search.py              : Stage 5+6 evidence retrieval and similarity ranking
- query_pipeline.py      : Stage 7+8 context augmentation and answer synthesis
- main.py                : Interactive demo CLI and Raw LLM vs Naive RAG scorecard
- Day4_Naive_RAG_Project_Build_Spec.md : 8-box architecture specification

ADK Multi-Agent System:
- coordinator/           : Root coordinator agent (routes to workers via sub_agents)
- code_helper/           : Top-level agent package for ADK Web & CLI discovery
- onboarding_guide/      : Onboarding guide agent
- architecture_mapper/   : Static AST codebase analysis agent
- code_searcher/         : Semantic RAG code search worker
- code_explainer/        : MCP Filesystem file reader and deep explainer
""".strip()


# ═══════════════════════════════════════════════════════════════════════════
# LiteLLM Single-Tool Adapter (prevents NVIDIA NIM 400 multi-tool error)
# ═══════════════════════════════════════════════════════════════════════════
class SingleToolCallLiteLlm(LiteLlm):
    """LiteLlm adapter enforcing single tool-call execution for NVIDIA NIM."""

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

# ═══════════════════════════════════════════════════════════════════════════
# Agent Definition
# ═══════════════════════════════════════════════════════════════════════════
_INSTRUCTION = """\
You are **Onboarding Guide**, an expert technical guide for the Naive RAG project.

## Your Capabilities & Tools:
1. `get_project_summary`: Call to get the 8 architecture pipeline stages and key files.
2. `get_setup_and_run_instructions`: Call with topic ('env', 'index', 'run', or 'learning') to get verified commands.
3. `list_project_files`: Call to get the list of primary project files.

## Workflow Rules:
- Call at most ONE tool that directly addresses the user's question.
- After receiving the tool output, immediately synthesize a clear, friendly, and complete markdown answer.
- Do NOT call additional tools once you have the needed information.
- Base your answers strictly on the tool outputs. Never invent commands or file names.
- Never output real API keys or sensitive secrets.
"""

root_agent = Agent(
    name="onboarding_guide",
    description=(
        "Onboards new developers by explaining what the project does, key files to read, "
        "safe environment setup, indexing commands, and how to run the application."
    ),
    model=_litellm_model,
    instruction=_INSTRUCTION,
    tools=[
        get_project_summary,
        get_setup_and_run_instructions,
        list_project_files,
    ],
)
