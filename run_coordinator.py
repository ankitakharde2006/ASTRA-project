"""
run_coordinator.py — Interactive and CLI Runner for the Coordinator Agent.

Usage:
    # Run a single query from the command line:
    python run_coordinator.py "How do I get started with this project?"
    python run_coordinator.py "What functions are defined in search.py?"
    python run_coordinator.py "Explain the config.py file in detail"
    python run_coordinator.py "Search the codebase for cosine similarity calculation"

    # Start an interactive chat session:
    python run_coordinator.py
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add project root to path
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from coordinator import root_agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types


def print_banner():
    banner = """
======================================================================
  ASTRATECH NAIVE RAG — ADK MULTI-AGENT COORDINATOR
======================================================================
  Connected Workers:
    1. onboarding_guide    — Project overview, 8 stages, setup, run
    2. architecture_mapper — AST static structure, files, imports, functions
    3. code_searcher       — RAG semantic vector search & evidence retrieval
    4. code_explainer      — Complete file explanations via MCP Filesystem
======================================================================
"""
    print(banner)


async def execute_query(runner: Runner, session_id: str, query: str):
    print(f"\n[User Question]: {query}")
    print("-" * 70)

    msg = types.Content(parts=[types.Part.from_text(text=query)], role="user")
    routed_worker = None
    tool_invocations = []
    response_text = ""
    event_count = 0
    max_events = 50  # Safety limit to prevent infinite loops

    try:
        async for event in runner.run_async(
            user_id="cli_user",
            session_id=session_id,
            new_message=msg,
        ):
            event_count += 1
            if event_count > max_events:
                print("\n[SAFETY] Maximum event limit reached. Stopping.")
                break

            if event.content and event.content.parts:
                for part in event.content.parts:
                    if part.function_call:
                        fc = part.function_call
                        if fc.name == "transfer_to_agent":
                            target = fc.args.get("agent_name", "")
                            routed_worker = target
                            print(f"[Coordinator Routing] -> Transferred to: @{target}")
                        else:
                            print(f"[{event.author}] Tool: {fc.name}({fc.args})")
                            tool_invocations.append((event.author, fc.name))
                    if part.text:
                        if event.author != "coordinator":
                            response_text += part.text
    except Exception as e:
        print(f"\n[ERROR] {type(e).__name__}: {e}")

    print("-" * 70)
    print(f"[Answered by Worker]: @{routed_worker or 'unknown'}")
    print(f"[Tool Calls]: {len(tool_invocations)}")
    print("-" * 70)
    print(response_text.strip() if response_text.strip() else "(No text response received)")
    print("=" * 70)
    return routed_worker, response_text


async def interactive_mode():
    print_banner()
    print("Type your question below (or 'exit' to quit).\n")
    runner = Runner(
        agent=root_agent,
        app_name="coordinator_cli",
        session_service=InMemorySessionService(),
    )
    session = await runner.session_service.create_session(
        app_name="coordinator_cli",
        user_id="cli_user",
    )

    while True:
        try:
            user_input = input("\nYou > ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "q"):
                print("\nGoodbye!")
                break
            await execute_query(runner, session.id, user_input)
        except (KeyboardInterrupt, EOFError):
            print("\nSession ended.")
            break


async def single_query_mode(query: str):
    print_banner()
    runner = Runner(
        agent=root_agent,
        app_name="coordinator_cli",
        session_service=InMemorySessionService(),
    )
    session = await runner.session_service.create_session(
        app_name="coordinator_cli",
        user_id="cli_user",
    )
    await execute_query(runner, session.id, query)


def main():
    parser = argparse.ArgumentParser(description="Run Google ADK Coordinator with sub-agents")
    parser.add_argument("query", nargs="?", default=None, help="Optional user query to run once")
    args = parser.parse_args()

    if args.query:
        asyncio.run(single_query_mode(args.query))
    else:
        asyncio.run(interactive_mode())


if __name__ == "__main__":
    main()
