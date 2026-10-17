"""
code_searcher — Google ADK agent that answers questions about indexed code
using the existing Naive RAG retrieval pipeline.

Architecture
------------
1.  The RAG search logic lives entirely in ``search.py`` (Stages 5+6).
    This agent wraps it as an ADK ``FunctionTool`` — no LangChain, no LlamaIndex.
2.  The LLM backing the agent is the NVIDIA chat model from ``config.py``,
    accessed through LiteLLM so ADK can manage the conversation loop.
3.  The agent's system instruction forces grounded answers: it must cite the
    retrieved evidence and refuse to guess when no useful chunks are found.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Make the parent Naive_RAG package importable so we can reuse search / config
# ---------------------------------------------------------------------------
_PARENT_DIR = str(Path(__file__).resolve().parent.parent)
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

from config import get_rag_config  # noqa: E402 — after sys.path fix
from google.adk import Agent  # noqa: E402
from google.adk.models import LiteLlm  # noqa: E402

# ---------------------------------------------------------------------------
# Load shared RAG configuration once at import time
# ---------------------------------------------------------------------------
_rag_config = get_rag_config()


# ---------------------------------------------------------------------------
# RAG tool — thin wrapper around the tested search() function
# ---------------------------------------------------------------------------
def rag_search(question: str, top_k: Optional[int] = None) -> str:
    """Search the indexed codebase for chunks relevant to ``question``.

    This tool embeds the question with NVIDIA NIM, queries the Pinecone
    vector store (or local fallback), and returns the top matching code
    chunks with similarity scores and source-file metadata.

    Args:
        question: Natural-language question about the codebase.
        top_k: Number of results to return (default from RAG config).

    Returns:
        A formatted string with the retrieved evidence chunks, including
        source file, chunk index, similarity score, relevance tier, and
        the chunk text.  Returns a "no results" message when the index
        has nothing relevant.
    """
    # Import lazily so the search module (and its heavy deps) load only when
    # the tool is actually invoked, not when the agent package is imported.
    from search import search, SearchError, SearchInputError

    try:
        results = search(
            question=question,
            config=_rag_config,
            top_k=top_k,
        )
    except SearchInputError as exc:
        return f"Invalid query: {exc}"
    except SearchError as exc:
        return f"Search failed: {exc}"

    if not results:
        return (
            "No relevant code chunks were found in the index for that question. "
            "The index may be empty, or the question may be unrelated to the "
            "indexed codebase."
        )

    # Format evidence for the LLM to consume
    lines: list[str] = [f"Retrieved {len(results)} evidence chunk(s):\n"]
    for r in results:
        chunk_label = (
            f"chunk {r['chunk_index']}" if r["chunk_index"] is not None else "—"
        )
        lines.append(
            f"--- Result #{r['rank']} [{r['relevance']}] "
            f"(score={r['score']:.4f}) ---\n"
            f"Source : {r['source_file'] or 'unknown'} / {chunk_label}\n"
            f"Text   :\n{r['text']}\n"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LiteLLM model — routes to NVIDIA NIM via the OpenAI-compatible endpoint
# ---------------------------------------------------------------------------
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
    max_tokens=1024,
    parallel_tool_calls=False,
)


# ---------------------------------------------------------------------------
# Agent definition
# ---------------------------------------------------------------------------
_INSTRUCTION = """\
You are **Code Searcher**, a technical assistant that answers questions about
an indexed codebase by retrieving and citing real code evidence.

## Workflow
1. When the user asks a question, call the **rag_search** tool with their
   question to retrieve relevant code chunks from the vector store.
2. Read every returned chunk carefully.
3. Compose your answer **strictly from the retrieved evidence**.
   - Quote or paraphrase the exact text from the chunks.
   - Cite the source file and chunk index for every fact you mention.

## Rules you MUST follow
- **Never invent** filenames, function names, class names, variable names,
  or implementation details that do not appear in the retrieved chunks.
- **Never guess** how code works beyond what the evidence shows.
- If the retrieved chunks do not contain enough information to answer the
  question, say exactly:
  "I could not find the answer in the retrieved code evidence."
- Do **not** call rag_search more than twice per question.
- Keep answers concise, factual, and well-structured.
"""

root_agent = Agent(
    name="code_searcher",
    description=(
        "Answers questions about an indexed codebase by retrieving real code "
        "chunks via RAG (Retrieval-Augmented Generation) and citing evidence."
    ),
    model=_litellm_model,
    instruction=_INSTRUCTION,
    tools=[rag_search],
)
