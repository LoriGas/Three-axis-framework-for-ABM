"""Regression tests for the random-founder, fixed-inheritance rule variant."""

import random
import unittest

import numpy as np
import torch

from config import RULE_VARIANT_TAGS, get_params_for_scenario
from model import World
from run_analysis import _build_tasks
from simulation import _resolve_rb_variant


class RuleRandomFixedTests(unittest.TestCase):
    def setUp(self) -> None:
        random.seed(321)
        np.random.seed(321)
        torch.manual_seed(321)

    def _world(self) -> World:
        params = get_params_for_scenario("S3")
        params.initial_agents = 12
        fixed_alpha, fixed_beta, mutate, resample, learn = _resolve_rb_variant(
            "rb_rand_fix"
        )
        self.assertIsNone(fixed_alpha)
        self.assertIsNone(fixed_beta)
        return World(
            params,
            agent_type="rule",
            mutate_propensity=mutate,
            random_offspring_propensity=resample,
            propensity_learning=learn,
        )

    def test_variant_is_registered_and_scheduled(self) -> None:
        self.assertIn("rb_rand_fix", RULE_VARIANT_TAGS)
        tasks = [task for task in _build_tasks() if task[2] == "Rule-Random/Fix"]
        self.assertEqual(len(tasks), 5)

    def test_founders_receive_heterogeneous_random_genes(self) -> None:
        world = self._world()
        genes = [(agent.alpha, agent.beta) for agent in world.agents]

        self.assertGreater(len(set(genes)), 1)
        self.assertTrue(all(0.0 <= alpha <= 2.0 for alpha, _ in genes))
        self.assertTrue(all(0.0 <= beta <= 1.0 for _, beta in genes))

    def test_genes_are_inherited_exactly_across_generations(self) -> None:
        world = self._world()
        parent = world.agents[0]
        occupied = {(agent.x, agent.y) for agent in world.agents}

        child = world._attempt_reproduction(parent, occupied, True)
        self.assertIsNotNone(child)
        self.assertEqual(child.alpha, parent.alpha)
        self.assertEqual(child.beta, parent.beta)

        grandchild = world._attempt_reproduction(child, occupied, True)
        self.assertIsNotNone(grandchild)
        self.assertEqual(grandchild.alpha, parent.alpha)
        self.assertEqual(grandchild.beta, parent.beta)


if __name__ == "__main__":
    unittest.main()
