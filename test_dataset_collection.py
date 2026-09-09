"""Regression tests for behavioural-cloning dataset collection."""

import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from config import get_params_for_scenario
from lib_training import DATASET_HEADER, _rollout_and_log_evo_top5
from model import World


class DatasetCollectionTests(unittest.TestCase):
    def setUp(self) -> None:
        random.seed(123)
        np.random.seed(123)
        torch.manual_seed(123)

    def test_world_exposes_the_exact_action_phase_records(self) -> None:
        params = get_params_for_scenario("S3")
        world = World(params, agent_type="rule", mutate_propensity=True)
        captured = []
        original = world._decide_actions_batch

        def capture(agents, percepts):
            actions = original(agents, percepts)
            captured.extend(
                (agent, list(percept), tuple(action))
                for agent, percept, action in zip(agents, percepts, actions)
            )
            return actions

        world._decide_actions_batch = capture
        world.step()

        self.assertEqual(len(world.decision_records_step), len(captured))
        for observed, expected in zip(world.decision_records_step, captured):
            self.assertIs(observed[0], expected[0])
            self.assertEqual(observed[1], expected[1])
            self.assertEqual(observed[2], expected[2])

    def test_top_agent_dataset_has_episode_groups_and_complete_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "dataset.csv"
            _rollout_and_log_evo_top5(
                "S3",
                output,
                min_rows=200,
                min_episodes=3,
                max_rows_per_episode=100,
                max_steps=60,
                min_survival=0,
            )
            frame = pd.read_csv(output)

        self.assertEqual(list(frame.columns), DATASET_HEADER)
        self.assertEqual(frame["episode_id"].nunique(), 3)
        self.assertLessEqual(int(frame.groupby("episode_id").size().max()), 100)
        self.assertTrue(set(frame["move_target"]).issubset(set(range(5))))
        self.assertTrue(set(frame["repro_target"]).issubset({0, 1}))


if __name__ == "__main__":
    unittest.main()
