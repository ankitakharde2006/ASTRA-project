"""
test_onboarding_guide.py — Verification suite for the Onboarding Guide worker.

Tests:
1. get_project_summary() returns the 8 pipeline stages and key files.
2. get_onboarding_guide_steps() returns verified commands for env, index, run, learning.
3. No secrets or API keys are leaked in tool outputs.
4. Agent instantiation, name, and registered tools.
"""

import unittest
from onboarding_guide import (
    get_project_summary,
    get_setup_and_run_instructions,
    root_agent,
)


class TestOnboardingGuide(unittest.TestCase):

    def test_01_project_summary(self):
        summary = get_project_summary()
        self.assertIn("AstraTech Naive RAG Pipeline", summary)
        self.assertIn("Stage 1: Document Ingestion", summary)
        self.assertIn("Stage 8: Grounded Generation", summary)
        self.assertIn("Day4_Naive_RAG_Project_Build_Spec.md", summary)
        self.assertIn("main.py", summary)

    def test_02_onboarding_steps(self):
        # Env topic
        env_info = get_setup_and_run_instructions("env")
        self.assertIn("NVIDIA_API_KEY", env_info)
        self.assertIn("PINECONE_API_KEY", env_info)
        self.assertIn("use_mock_ai", env_info)

        # Index topic
        index_info = get_setup_and_run_instructions("index")
        self.assertIn("python indexer.py", index_info)
        self.assertIn("python main.py --index", index_info)

        # Run topic
        run_info = get_setup_and_run_instructions("run")
        self.assertIn("python main.py", run_info)
        self.assertIn("python main.py --compare", run_info)
        self.assertIn("python main.py --mock", run_info)

        # Learning topic
        learn_info = get_setup_and_run_instructions("learning")
        self.assertIn("python main.py --compare", learn_info)
        self.assertIn("adk run", learn_info)

    def test_03_no_secret_leaks(self):
        combined = get_project_summary() + get_setup_and_run_instructions()
        # Ensure no real API key prefix or live credentials are leaked
        self.assertNotIn("nvapi-", combined)
        self.assertNotIn("pcsk_", combined)

    def test_04_agent_registration(self):
        self.assertEqual(root_agent.name, "onboarding_guide")
        tool_names = [t.__name__ for t in root_agent.tools]
        self.assertIn("get_project_summary", tool_names)
        self.assertIn("get_setup_and_run_instructions", tool_names)
        self.assertIn("list_project_files", tool_names)
        self.assertEqual(len(tool_names), 3)


if __name__ == "__main__":
    unittest.main()
