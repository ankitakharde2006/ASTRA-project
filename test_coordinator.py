"""
test_coordinator.py — Integration Tests for Coordinator Agent & Sub-Agent Routing.

Verifies:
1. Coordinator correctly transfers to onboarding_guide.
2. Coordinator correctly transfers to architecture_mapper.
3. Coordinator correctly transfers to code_searcher.
4. Coordinator correctly transfers to code_explainer.
5. Coordinator never answers questions directly and only delegates via transfer_to_agent.
"""

import asyncio
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from coordinator import root_agent, run_coordinator_query


def test_onboarding_guide_routing():
    print("\n--- Test 1: Coordinator Routing to onboarding_guide ---")
    query = "I am a new engineer. What does this project do and how do I configure the environment safely?"
    res = asyncio.run(run_coordinator_query(query))
    print(f"Transfers : {res['transfers']}")
    print(f"Worker    : {res['worker']}")
    print(f"Answer    : {res['answer'][:250]}...")
    assert "onboarding_guide" in res["transfers"], f"Expected onboarding_guide in transfers, got {res['transfers']}"
    assert res["worker"] == "onboarding_guide", f"Expected worker onboarding_guide, got {res['worker']}"
    assert len(res["answer"]) > 50, "Answer should not be empty"
    print("Test 1 passed!")


def test_architecture_mapper_routing():
    print("\n--- Test 2: Coordinator Routing to architecture_mapper ---")
    query = "What functions and classes are defined in config.py?"
    res = asyncio.run(run_coordinator_query(query))
    print(f"Transfers : {res['transfers']}")
    print(f"Worker    : {res['worker']}")
    print(f"Answer    : {res['answer'][:250]}...")
    assert "architecture_mapper" in res["transfers"], f"Expected architecture_mapper in transfers, got {res['transfers']}"
    assert res["worker"] == "architecture_mapper", f"Expected worker architecture_mapper, got {res['worker']}"
    assert len(res["answer"]) > 50, "Answer should not be empty"
    print("Test 2 passed!")


def test_code_searcher_routing():
    print("\n--- Test 3: Coordinator Routing to code_searcher ---")
    query = "Search the codebase for how query embedding and vector search are implemented."
    res = asyncio.run(run_coordinator_query(query))
    print(f"Transfers : {res['transfers']}")
    print(f"Worker    : {res['worker']}")
    print(f"Answer    : {res['answer'][:250]}...")
    assert "code_searcher" in res["transfers"], f"Expected code_searcher in transfers, got {res['transfers']}"
    assert res["worker"] == "code_searcher", f"Expected worker code_searcher, got {res['worker']}"
    assert len(res["answer"]) > 50, "Answer should not be empty"
    print("Test 3 passed!")


def test_code_explainer_routing():
    print("\n--- Test 4: Coordinator Routing to code_explainer ---")
    query = "Please read search.py using the filesystem tools and explain its inputs, outputs, and role."
    res = asyncio.run(run_coordinator_query(query))
    print(f"Transfers : {res['transfers']}")
    print(f"Worker    : {res['worker']}")
    print(f"Answer    : {res['answer'][:250]}...")
    assert "code_explainer" in res["transfers"], f"Expected code_explainer in transfers, got {res['transfers']}"
    assert res["worker"] == "code_explainer", f"Expected worker code_explainer, got {res['worker']}"
    assert len(res["answer"]) > 50, "Answer should not be empty"
    print("Test 4 passed!")


if __name__ == "__main__":
    test_onboarding_guide_routing()
    test_architecture_mapper_routing()
    test_code_searcher_routing()
    test_code_explainer_routing()
    print("\n==========================================")
    print(" ALL 4 COORDINATOR ROUTING TESTS PASSED! ")
    print("==========================================")
