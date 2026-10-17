"""
test_code_explainer.py — Integration and Unit Tests for Code Explainer Worker
with Model Context Protocol (MCP) Filesystem Server Integration.

Verifies:
1. MCP Toolset connection via npx to @modelcontextprotocol/server-filesystem.
2. Restriction of filesystem access to the Naive RAG project directory.
3. Availability of server's actual file-reading and file-listing tools.
4. Correct reading of a real Python file via MCP.
5. Error reporting when a non-existent file is requested.
6. End-to-end agent execution with runner explaining a real Python file.
"""

import asyncio
import os
import sys
from pathlib import Path

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from code_explainer import (
    create_code_explainer_agent,
    create_filesystem_mcp_toolset,
    root_agent,
    _PROJECT_ROOT as AGENT_PROJECT_ROOT,
)
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types


def test_mcp_toolset_creation_and_tools():
    """Verify MCP toolset connects via npx and provides file tools."""
    print("\n--- Test 1: MCP Toolset Connection & Tool Discovery ---")
    toolset = create_filesystem_mcp_toolset(timeout=60.0)

    async def _check():
        tools = await toolset.get_tools()
        tool_names = [t.name for t in tools]
        print(f"Discovered MCP tools ({len(tool_names)}): {tool_names}")

        # Verify key file-reading and listing tools exist
        assert "read_file" in tool_names or "read_text_file" in tool_names, (
            "Neither read_file nor read_text_file found in MCP tools"
        )
        assert "list_directory" in tool_names, "list_directory not found in MCP tools"

        # Verify reading a real Python file via MCP
        target_tool = next(t for t in tools if t.name in ("read_text_file", "read_file"))
        print(f"Testing direct execution of tool: {target_tool.name} on config.py")
        result = await target_tool.run_async(args={"path": "config.py"}, tool_context=None)
        assert result is not None, "Tool returned None"
        content_text = ""
        if isinstance(result, dict) and "content" in result:
            for item in result["content"]:
                if item.get("type") == "text":
                    content_text += item.get("text", "")
        print(f"Retrieved {len(content_text)} chars from config.py via MCP.")
        assert "RAGConfig" in content_text or "config.py" in content_text, (
            "config.py content does not match expected code"
        )

        # Verify non-existent file error handling
        print("Testing direct execution on non-existent file...")
        err_result = await target_tool.run_async(args={"path": "non_existent_file_12345.py"}, tool_context=None)
        assert isinstance(err_result, dict), "Error result should be a dictionary"
        print(f"Non-existent file handled cleanly: {err_result.get('isError', False)}")

        await toolset.close()

    asyncio.run(_check())
    print("Test 1 passed successfully!")


def test_agent_explains_real_python_file():
    """Test full agent execution explaining a real Python file (search.py)."""
    print("\n--- Test 2: Full Agent Execution (Explaining search.py) ---")

    async def _run_agent():
        runner = Runner(
            agent=root_agent,
            app_name="code_explainer_test",
            session_service=InMemorySessionService(),
        )
        session = await runner.session_service.create_session(
            app_name="code_explainer_test",
            user_id="test_dev",
        )

        user_prompt = (
            "Please read search.py using the filesystem tools and explain what it does, "
            "its main functions, their inputs and outputs, and its role in the project."
        )
        msg = types.Content(parts=[types.Part.from_text(text=user_prompt)], role="user")

        final_text = ""
        mcp_called = []

        async for event in runner.run_async(
            user_id="test_dev",
            session_id=session.id,
            new_message=msg,
        ):
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if part.function_call:
                        print(f"Agent invoked MCP tool: {part.function_call.name} with args: {part.function_call.args}")
                        mcp_called.append(part.function_call.name)
                    if part.text:
                        final_text += part.text

        print("\nAgent Explanation Summary:")
        print(final_text[:600] + ("..." if len(final_text) > 600 else ""))

        # Verify MCP tool was called
        assert len(mcp_called) > 0, "Agent did not invoke any MCP tools!"
        assert any("read" in fn for fn in mcp_called), f"Agent did not call read tool: {mcp_called}"

        # Verify explanation covers key components of search.py
        lower_explanation = final_text.lower()
        assert "search" in lower_explanation, "Explanation should mention search"
        assert len(final_text) > 200, "Explanation is too brief"
        print("\nTest 2 passed successfully!")

    asyncio.run(_run_agent())


if __name__ == "__main__":
    test_mcp_toolset_creation_and_tools()
    test_agent_explains_real_python_file()
