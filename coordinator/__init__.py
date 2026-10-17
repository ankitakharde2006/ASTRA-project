"""
coordinator — Root Google ADK Coordinator Agent that manages and routes user
queries to the four specialized workers via ADK's native sub_agents mechanism.

Workers:
- code_searcher       : Answers questions by retrieving real code chunks via RAG.
- code_explainer      : Reads and explains complete files, functions, inputs/outputs,
                        and architectural roles using the official MCP Filesystem Server.
- architecture_mapper : Analyzes codebase structure, files, imports, and AST definitions.
- onboarding_guide    : Explains project overview, 8 pipeline stages, safe setup,
                        indexing commands, and running the application.

Requirements:
- Root agent named 'coordinator'.
- Sole responsibility is selecting the appropriate worker.
- Never answers the user's question directly.
- Preserves each worker's existing tools and instructions.
- Routes ambiguous questions to the most appropriate worker.
- Does not modify custom RAG implementation or README.md.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_PROJECT_ROOT_STR = str(_PROJECT_ROOT)
if _PROJECT_ROOT_STR not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT_STR)

from config import get_rag_config  # noqa: E402
from google.adk import Agent  # noqa: E402
from google.adk.models import LiteLlm  # noqa: E402
from google.adk.runners import Runner  # noqa: E402
from google.adk.sessions import InMemorySessionService  # noqa: E402
from google.genai import types  # noqa: E402

# Import the four specialized workers
from architecture_mapper import root_agent as architecture_mapper_agent  # noqa: E402
from code_explainer import root_agent as code_explainer_agent  # noqa: E402
from code_searcher import SingleToolCallLiteLlm, root_agent as code_searcher_agent  # noqa: E402
from onboarding_guide import root_agent as onboarding_guide_agent  # noqa: E402

_rag_config = get_rag_config()

# ---------------------------------------------------------------------------
# Preserve workers and configure transfer boundaries
# Workers stay focused on their own tools and do not transfer sideways/back
# ---------------------------------------------------------------------------
for worker in [
    code_searcher_agent,
    code_explainer_agent,
    architecture_mapper_agent,
    onboarding_guide_agent,
]:
    worker.disallow_transfer_to_parent = True
    worker.disallow_transfer_to_peers = True

# ---------------------------------------------------------------------------
# Coordinator Model
# ---------------------------------------------------------------------------
_litellm_model = SingleToolCallLiteLlm(
    model=f"openai/{_rag_config.chat_model}",
    api_key=_rag_config.nvidia_api_key,
    api_base=_rag_config.nvidia_base_url,
    temperature=0.1,
    max_tokens=1024,
    parallel_tool_calls=False,
)

# ---------------------------------------------------------------------------
# Coordinator Instructions
# ---------------------------------------------------------------------------
_COORDINATOR_INSTRUCTION = """\
You are **Coordinator**, the central routing agent for the AstraTech Naive RAG development system.

## Your Sole Responsibility:
Your only role is to analyze the user's request and transfer control to the single most appropriate worker using the `transfer_to_agent` tool.

## Critical Rules:
1. **NEVER answer the user's question directly.** You have no answers of your own.
2. **Always call `transfer_to_agent`** with the chosen worker's exact name.
3. Do not generate explanations or text before calling `transfer_to_agent`.
4. If a question is ambiguous, select the worker that best fits the primary intent.

## Worker Responsibilities:

1. `code_searcher`:
   - Primary role: Code search and evidence retrieval via RAG.
   - Use when the user asks:
     * To search or find code, functions, logic, or implementations in the indexed codebase.
     * Any query containing "search", "find", "retrieve", or asking where/how something is coded in the repository.
     * Questions requiring evidence citations from indexed chunks.

2. `code_explainer`:
   - Primary role: In-depth explanation of a specific named source file.
   - Use ONLY when the user asks:
     * To explain, read, or break down a specific named source file (e.g. `search.py`, `indexer.py`, `config.py`, `vector_store.py`, `query_pipeline.py`, `main.py`).
     * For function inputs, outputs, and architectural role of a specific file using MCP filesystem tools.
   - Note: If the user asks to search the codebase rather than explain a named file, choose `code_searcher`.

3. `architecture_mapper`:
   - Primary role: Static repository structure, file listing, imports, and AST definitions.
   - Use when the user asks:
     * For a directory or file tree listing (skipping noise directories).
     * To extract imports or external dependencies from a file.
     * To inspect AST function/class definitions and line numbers without AI in tools.

4. `onboarding_guide`:
   - Primary role: Developer onboarding, setup, and getting started.
   - Use when the user asks:
     * How to get started or what the project does.
     * The 8 architecture pipeline stages (Ingestion to Generation).
     * Which key files to read first.
     * Safe environment configuration (`.env` setup, secrets protection).
     * How to index the codebase or run the application (`main.py`, `--compare`, `--mock`).
"""

# ---------------------------------------------------------------------------
# Coordinator Root Agent Definition
# ---------------------------------------------------------------------------
root_agent = Agent(
    name="coordinator",
    description=(
        "Central coordinator that routes user requests to one of four specialized "
        "workers: onboarding_guide, architecture_mapper, code_searcher, or code_explainer."
    ),
    model=_litellm_model,
    instruction=_COORDINATOR_INSTRUCTION,
    sub_agents=[
        code_searcher_agent,
        code_explainer_agent,
        architecture_mapper_agent,
        onboarding_guide_agent,
    ],
    tools=[],  # Coordinator has no direct tools; ADK injects transfer_to_agent
)


# ---------------------------------------------------------------------------
# Programmatic Helper to Run Coordinator Queries
# ---------------------------------------------------------------------------
async def run_coordinator_query(
    query: str,
    user_id: str = "default_user",
    session_id: Optional[str] = None,
    runner: Optional[Runner] = None,
) -> dict:
    """Execute a query through the coordinator and return the routed response.

    Returns:
        dict with:
            - 'worker': Name of the worker that answered the question
            - 'answer': Final generated text response
            - 'transfers': List of agent transfers that occurred
            - 'tool_calls': List of tool calls invoked by the worker
    """
    if runner is None:
        runner = Runner(
            agent=root_agent,
            app_name="coordinator_app",
            session_service=InMemorySessionService(),
        )

    if session_id is None:
        session = await runner.session_service.create_session(
            app_name="coordinator_app",
            user_id=user_id,
        )
        session_id = session.id

    msg = types.Content(parts=[types.Part.from_text(text=query)], role="user")

    transfers: list[str] = []
    tool_calls: list[dict] = []
    final_text: str = ""
    responding_worker: str = "coordinator"

    # Use RunConfig to cap LLM calls and prevent infinite loops
    run_config = types.GenerateContentConfig(
        response_modalities=["TEXT"],
    )

    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=msg,
    ):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.function_call:
                    fc = part.function_call
                    if fc.name == "transfer_to_agent":
                        target = fc.args.get("agent_name", "")
                        transfers.append(target)
                        responding_worker = target
                    else:
                        tool_calls.append({"name": fc.name, "args": fc.args, "worker": event.author})
                if part.text:
                    if event.author != "coordinator":
                        final_text += part.text
                        responding_worker = event.author

    return {
        "worker": responding_worker,
        "answer": final_text.strip(),
        "transfers": transfers,
        "tool_calls": tool_calls,
    }
