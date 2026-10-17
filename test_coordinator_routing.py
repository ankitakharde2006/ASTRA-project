"""
test_coordinator_routing.py — Verify coordinator routing across all 4 workers.

Verifies:
1. Coordinator selects onboarding_guide for onboarding/setup questions.
2. Coordinator selects architecture_mapper for static AST/file structure questions.
3. Coordinator selects code_searcher for codebase search/RAG evidence retrieval.
4. Coordinator selects code_explainer for reading and explaining specific files.
5. Coordinator never answers directly (its output is exclusively transfer_to_agent).
"""

import asyncio
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from coordinator import root_agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types


async def get_coordinator_routing_decision(query: str) -> str:
    """Run one turn of coordinator and return the sub-agent it transferred to."""
    runner = Runner(
        agent=root_agent,
        app_name="routing_verifier",
        session_service=InMemorySessionService(),
    )
    session = await runner.session_service.create_session(
        app_name="routing_verifier",
        user_id="test_user",
    )
    msg = types.Content(parts=[types.Part.from_text(text=query)], role="user")

    transferred_to = None
    coordinator_text = ""

    async for event in runner.run_async(
        user_id="test_user",
        session_id=session.id,
        new_message=msg,
    ):
        if event.author == "coordinator" and event.content and event.content.parts:
            for part in event.content.parts:
                if part.function_call and part.function_call.name == "transfer_to_agent":
                    transferred_to = part.function_call.args.get("agent_name")
                if part.text:
                    coordinator_text += part.text

        # Once transfer has been initiated, we have our routing decision
        if transferred_to:
            break

    # Verify coordinator never answers directly
    assert not coordinator_text.strip(), f"Coordinator must never answer directly! Got text: {coordinator_text}"
    return transferred_to


def test_all_routes():
    test_cases = [
        (
            "I am a new developer on the team. What does this project do and how do I set up my .env file?",
            "onboarding_guide",
        ),
        (
            "List all Python source files in the project and extract imports from config.py",
            "architecture_mapper",
        ),
        (
            "Search the indexed vector store for how the cosine similarity ranking works.",
            "code_searcher",
        ),
        (
            "Please read and deeply explain search.py, its functions, inputs, and outputs using the filesystem.",
            "code_explainer",
        ),
    ]

    for query, expected_worker in test_cases:
        print(f"\nQuery: \"{query}\"")
        target = asyncio.run(get_coordinator_routing_decision(query))
        print(f"  -> Coordinator Routed To: @{target} (Expected: @{expected_worker})")
        assert target == expected_worker, f"Expected routing to {expected_worker}, but got {target}"

    print("\n" + "=" * 50)
    print(" ALL 4 WORKER ROUTING TESTS PASSED! ")
    print("=" * 50)


if __name__ == "__main__":
    test_all_routes()
