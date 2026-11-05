"""Regression tests for LLM decision logging and episode rejection."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agents.llm_agent import LLMAgent, LLMDecisionClient
from config import get_params_for_scenario
from run_analysis_llm import _run_episode


class LLMQualityControlTests(unittest.TestCase):
    def tearDown(self) -> None:
        LLMAgent._client = None

    def test_unavailable_client_returns_auditable_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ,
            {
                "LLM_ENABLED": "0",
                "LLM_LOG_FILE": str(Path(directory) / "calls.jsonl"),
            },
            clear=False,
        ):
            client = LLMDecisionClient()
            params = get_params_for_scenario("S3")
            result = client.decide([0.0] * 9 + [2.0], params)

        self.assertEqual(result, (4, False, "client_unavailable", False))

    def test_failed_decisions_reject_episode_before_first_valid_step(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ,
            {
                "LLM_ENABLED": "0",
                "LLM_LOG_FILE": str(Path(directory) / "calls.jsonl"),
            },
            clear=False,
        ):
            LLMAgent._client = None
            params = get_params_for_scenario("S3")
            params.initial_agents = 5
            result = _run_episode(params, "S3", episode_idx=0, attempt=0)

        self.assertEqual(result.n_steps_alive, 0)
        self.assertGreater(result.technical_failures, 0)
        self.assertEqual(result.decisions, [])

if __name__ == "__main__":
    unittest.main()
