"""Focused regression tests for online-learning transitions and rewards."""

import random
import unittest

import numpy as np
import torch

from agents.neural import NeuralAgent
from config import get_params_for_scenario
from model import World


class OnlineLearningTests(unittest.TestCase):
    def setUp(self) -> None:
        random.seed(123)
        np.random.seed(123)
        torch.manual_seed(123)

    def test_reward_separates_movement_from_reproduction(self) -> None:
        params = get_params_for_scenario("S3")
        params.initial_agents = 1
        world = World(params, "random")

        energy_without_birth, fitness_without_birth = world._local_rewards(
            harvested_food=2.0,
            movement_cost=0.4,
            had_offspring=False,
        )
        energy_with_birth, fitness_with_birth = world._local_rewards(
            harvested_food=2.0,
            movement_cost=0.4,
            had_offspring=True,
        )

        self.assertEqual(energy_without_birth, energy_with_birth)
        self.assertGreater(fitness_with_birth, fitness_without_birth)

        _, terminal_with_birth = world._local_rewards(
            harvested_food=0.0,
            movement_cost=0.4,
            had_offspring=True,
            survived=False,
        )
        _, terminal_without_birth = world._local_rewards(
            harvested_food=0.0,
            movement_cost=0.4,
            had_offspring=False,
            survived=False,
        )
        self.assertEqual(terminal_with_birth, params.rl_birth_bonus)
        self.assertEqual(terminal_without_birth, 0.0)

    def test_nonterminal_partial_rollout_can_be_flushed(self) -> None:
        params = get_params_for_scenario("S3")
        agent = NeuralAgent(1, 1, 3.0, 1, online_learning=True, n_steps=4)
        percept = [1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 1.0, 3.0]
        before = [p.detach().clone() for p in agent.net.parameters()]

        agent.decide_action(percept, params)
        agent.update_online(-0.2, 0.02, next_percept=percept, done=False)
        agent.flush_online()

        delta = sum(
            float((after.detach() - old).abs().sum())
            for after, old in zip(agent.net.parameters(), before)
        )
        self.assertGreater(delta, 0.0)
        self.assertEqual(agent._traj, [])

    def test_terminal_transition_updates_and_clears_rollout(self) -> None:
        params = get_params_for_scenario("S3")
        agent = NeuralAgent(1, 1, 3.0, 1, online_learning=True, n_steps=4)
        percept = [1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 1.0, 3.0]
        before = [p.detach().clone() for p in agent.net.parameters()]

        agent.decide_action(percept, params)
        agent.update_online(-1.0, 0.0, next_percept=None, done=True)

        delta = sum(
            float((after.detach() - old).abs().sum())
            for after, old in zip(agent.net.parameters(), before)
        )
        self.assertGreater(delta, 0.0)
        self.assertEqual(agent._traj, [])

    def test_world_learns_from_transition_for_agent_dying_next_step(self) -> None:
        params = get_params_for_scenario("S3")
        params.initial_agents = 1
        params.initial_beta = 0.0
        params.base_metabolism = 0.01
        world = World(
            params,
            "rule",
            mutate_propensity=False,
            propensity_learning=True,
        )
        world.food_grid[:] = params.max_food_cell
        world.step()  # creates the pending transition
        self.assertEqual(len(world.agents), 1)

        world.agents[0].energy = 0.01
        world.food_grid[:] = 0.0
        world.params.base_metabolism = 1.0
        world.step()

        self.assertEqual(len(world.agents), 0)
        self.assertEqual(world.reward_events_step, 1)
        self.assertLess(world.total_reward_step, 0.0)

    def test_stochastic_frozen_policy_samples_without_updating(self) -> None:
        params = get_params_for_scenario("S3")
        agent = NeuralAgent(
            1, 1, 3.0, 1,
            online_learning=False,
            stochastic_policy=True,
        )
        percept = [1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 1.0, 3.0]
        before = [p.detach().clone() for p in agent.net.parameters()]

        for _ in range(10):
            agent.decide_action(percept, params)

        self.assertIsNone(agent._last_percept)
        self.assertFalse(hasattr(agent, "optimizer"))
        self.assertTrue(all(torch.equal(after, old) for after, old in zip(agent.net.parameters(), before)))

    def test_world_does_not_measure_reward_without_online_learning(self) -> None:
        params = get_params_for_scenario("S3")
        params.initial_agents = 1
        world = World(params, "random")
        world.food_grid[:] = params.max_food_cell

        world.step()
        world.step()

        self.assertEqual(world.reward_events_step, 0)
        self.assertEqual(world.total_reward_step, 0.0)
        self.assertIsNone(getattr(world.agents[0], "_pending_update", None))

    def test_world_still_measures_reward_with_online_learning(self) -> None:
        params = get_params_for_scenario("S3")
        params.initial_agents = 1
        world = World(
            params,
            "rule",
            mutate_propensity=False,
            propensity_learning=True,
        )
        world.food_grid[:] = params.max_food_cell

        world.step()  # creates a pending learning transition
        world.food_grid[:] = params.max_food_cell
        world.step()  # finalizes it at the next nutrition phase

        self.assertGreater(world.reward_events_step, 0)


if __name__ == "__main__":
    unittest.main()
