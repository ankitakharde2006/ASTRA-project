"""
code_explainer — Google ADK worker that explains codebase files using the
official Model Context Protocol (MCP) Filesystem Server.

Requirements:
- Connects to @modelcontextprotocol/server-filesystem through npx via Stdio.
- Restricts filesystem access exclusively to the Naive RAG project directory.
- Attaches the MCP toolset directly to the code_explainer worker.
- Uses the server's actual available file-listing and file-reading tool names
  (read_file, read_text_file, list_directory, get_file_info, directory_tree).
- Never uses a custom Python file-reading substitute.
- Strictly protects secrets: refuses to expose or read .env or credential files.
- Thoroughly explains complete files, their main functions, inputs, outputs,
  and role in the 8-stage Naive RAG pipeline.
- Clearly reports missing or unreadable files without hallucinating.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Project directory & path setup
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_PROJECT_ROOT_STR = str(_PROJECT_ROOT)
if _PROJECT_ROOT_STR not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT_STR)

from config import get_rag_config  # noqa: E402
from google.adk import Agent  # noqa: E402
from google.adk.models import LiteLlm  # noqa: E402
from google.adk.tools import McpToolset  # noqa: E402
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams  # noqa: E402
from mcp.client.stdio import StdioServerParameters  # noqa: E402

_rag_config = get_rag_config()


# ---------------------------------------------------------------------------
# LiteLLM adapter enforcing single tool-call per turn for NVIDIA NIM
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


# ---------------------------------------------------------------------------
# MCP Filesystem Server Toolset Builder
# ---------------------------------------------------------------------------
# Server's actual available file-listing and file-reading tool names
_ALLOWED_MCP_TOOLS = [
    "read_file",
    "read_text_file",
    "read_multiple_files",
    "list_directory",
    "list_directory_with_sizes",
    "get_file_info",
    "directory_tree",
    "list_allowed_directories",
]


def create_filesystem_mcp_toolset(
    project_dir: Optional[str | Path] = None,
    timeout: float = 60.0,
) -> McpToolset:
    """Create an ADK McpToolset connected to @modelcontextprotocol/server-filesystem via npx.

    Args:
        project_dir: Root directory path to restrict filesystem access to.
            Defaults to the Naive RAG project root.
        timeout: Subprocess ready timeout in seconds (default 60s for Node startup).

    Returns:
        Configured McpToolset ready to attach to an ADK Agent.
    """
    target_dir = str(Path(project_dir or _PROJECT_ROOT).resolve())
    npx_cmd = shutil.which("npx") or "npx"

    server_params = StdioServerParameters(
        command=npx_cmd,
        args=["-y", "@modelcontextprotocol/server-filesystem", target_dir],
    )
    connection_params = StdioConnectionParams(
        server_params=server_params,
        timeout=timeout,
    )

    return McpToolset(
        connection_params=connection_params,
        tool_filter=_ALLOWED_MCP_TOOLS,
    )


# ---------------------------------------------------------------------------
# System Instruction for Code Explainer
# ---------------------------------------------------------------------------
_INSTRUCTION = """\
You are **Code Explainer**, a senior software engineer assistant specialized in
explaining source code for the AstraTech Naive RAG project.

You have direct access to the project's filesystem via the official Model Context
Protocol (MCP) Filesystem Server tools.

## Available MCP Tools:
- `read_text_file` / `read_file`: Read the full text contents of a project file.
- `list_directory`: List all files and directories in a given folder.
- `get_file_info`: Inspect metadata (size, timestamps, type) of a file or directory.
- `directory_tree`: View the directory hierarchy as a recursive tree.

## Security & Secrets Protection (CRITICAL):
- **NEVER inspect, read, or print the contents of `.env` or any file storing API keys, passwords, or secrets.**
- If a user requests to read `.env` or secret keys, refuse politely and explain that `.env` contains confidential credentials that must not be exposed.
- If you encounter API keys or tokens anywhere, redact them immediately.
- Filesystem access is strictly restricted to the project root directory. Do not attempt to access directories outside the project root.

## Workflow:
1. When asked to explain a file or code component:
   - Call `read_text_file` or `read_file` with the requested file's path.
   - Do NOT guess or hallucinate the file content.
2. If the tool reports that the file does not exist (e.g., ENOENT error) or cannot be read:
   - Clearly report to the user that the file does not exist or could not be read.
   - Offer to check available files using `list_directory`.
3. When the file content is retrieved, provide a comprehensive explanation structured as:
   - **Role in the Project**: How this file fits into the 8-stage Naive RAG architecture (e.g. Ingestion, Chunking, Embedding, Storage, Retrieval, Context Injection, Generation) or supporting infrastructure.
   - **Main Classes & Functions**: Clear breakdown of each class and function defined in the file.
   - **Inputs & Outputs**: For each key function, list the arguments (types, defaults, meaning) and return values.
   - **Key Logic & Edge Cases**: Explain noteworthy implementation details, such as offline simulation mode, error handling, or vector database interactions.
4. Keep the explanation accurate, factual, and strictly grounded in the file content read via MCP.
5. Do NOT call non-existent tools like `summarize` or `explain`. Provide your final response in clear markdown text.
"""


# ---------------------------------------------------------------------------
# Agent Factory and Default Root Agent
# ---------------------------------------------------------------------------
def create_code_explainer_agent(
    model: Optional[LiteLlm] = None,
    toolset: Optional[McpToolset] = None,
) -> Agent:
    """Build and return a configured code_explainer Agent."""
    if model is None:
        model = SingleToolCallLiteLlm(
            model=f"openai/{_rag_config.chat_model}",
            api_key=_rag_config.nvidia_api_key,
            api_base=_rag_config.nvidia_base_url,
            temperature=0.1,
            max_tokens=2048,
            parallel_tool_calls=False,
        )

    if toolset is None:
        toolset = create_filesystem_mcp_toolset()

    return Agent(
        name="code_explainer",
        description=(
            "Inspects, reads, and explains codebase files, functions, inputs, "
            "outputs, and architectural roles in the Naive RAG project using "
            "the official Model Context Protocol (MCP) Filesystem Server."
        ),
        model=model,
        instruction=_INSTRUCTION,
        tools=[toolset],
    )


# Module-level default agent instance
root_agent = create_code_explainer_agent()
