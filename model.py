"""Simulation world: grid, agents, and per-step logic."""

import random
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Set, Tuple

import numpy as np

from config import (
    ACTIONS,
    ALPHA_MAX,
    ALPHA_MIN,
    BETA_MAX,
    BETA_MIN,
    MLP_EVOLUTION,
    MLP_OFFLINE_INIT,
    MLP_ONLINE_LEARNING,
    MLP_RANDOM_OFFSPRING,
    NEIGHBOR_OFFSETS,
    ModelParams,
)
from agents.base import AgentBase
from agents.llm_agent import LLMAgent
from agents.neural import NeuralAgent, mask_logits_invalid_numpy
from agents.random_agent import RandomAgent
from agents.rule_based import RuleBasedAgent
from network import NeuralNetwork


class World:
    """Simulation environment: resource grid + agents + one-step logic."""

    def __init__(
        self,
        params: ModelParams,
        agent_type: str = "rule",
        net_template: Optional[NeuralNetwork] = None,
        mutate_propensity: bool = True,
        random_offspring_propensity: bool = False,
        propensity_learning: bool = False,
        net_learning: bool = False,
    ) -> None:
        self.params = params
        self.agent_type = agent_type
        self.net_template = net_template
        self.mutate_offspring_net = agent_type in MLP_EVOLUTION
        self.random_offspring_net = agent_type in MLP_RANDOM_OFFSPRING
        self.propensity_learning = propensity_learning
        self.net_learning = net_learning or (agent_type in MLP_ONLINE_LEARNING)
        self.mutate_propensity = mutate_propensity
        self.random_offspring_propensity = random_offspring_propensity

        w, h = params.grid_width, params.grid_height
        self.food_grid: np.ndarray = np.random.uniform(0, params.max_food_cell, size=(w, h))
        self.agents: List[AgentBase] = []
        self.total_food: float = float(self.food_grid.sum())
        self.moves_step: int = 0
        self.births_step: int = 0
        self.deaths_step: int = 0
        self.harvested_food_step: float = 0.0
        self.action_counts_step: List[int] = [0, 0, 0, 0, 0]  # N, S, E, W, Stay
        self.total_reward_step: float = 0.0
        self.reward_events_step: int = 0
        self.total_agents_step: int = 0
        self.parent_ids_step: List[int] = []
        # Exact percept/action pairs produced in the current action phase.
        # Dataset generation consumes these records after ``step()`` so that
        # training data match the decisions used by the simulation.
        self.decision_records_step: List[
            Tuple[AgentBase, List[float], Tuple[int, int, bool]]
        ] = []

        self._occupied: Set[Tuple[int, int]] = set()
        self._init_agents()

    # ------------------------------------------------------------------
    # Agent factory
    # ------------------------------------------------------------------

    def _create_agent(
        self,
        x: int,
        y: int,
        energy: float,
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
    ) -> AgentBase:
        p = self.params
        if self.agent_type == "rule":
            a = alpha if alpha is not None else p.initial_alpha
            b = beta if beta is not None else p.initial_beta
            return RuleBasedAgent(
                x, y, energy, a,
                beta=b,
                online_learning=self.propensity_learning,
                lr_alpha=p.rl_lr_alpha,
                lr_beta=p.rl_lr_beta,
                baseline_alpha=p.rl_baseline_alpha,
                reward_clip=p.rl_reward_clip,
            )
        if self.agent_type == "random":
            return RandomAgent(x, y, energy)
        if self.agent_type == "llm":
            return LLMAgent(x, y, energy)
        if "mlp" in self.agent_type:
            tpl = self.net_template if self.agent_type in MLP_OFFLINE_INIT else None
            return NeuralAgent(
                x, y, energy,
                num_layers=p.net_hidden_layers,
                net_template=tpl,
                online_learning=self.net_learning,
                stochastic_policy=self.net_learning,
                lr=p.rl_lr_net,
                entropy_beta=p.rl_entropy_beta,
                critic_coef=p.rl_critic_coef,
                n_steps=p.rl_n_steps,
                gamma=p.rl_gamma,
            )
        raise ValueError(f"Unknown agent type: {self.agent_type}")

    def _init_agents(self) -> None:
        positions = [
            (x, y)
            for x in range(self.params.grid_width)
            for y in range(self.params.grid_height)
        ]
        random.shuffle(positions)
        n = min(self.params.initial_agents, len(positions))
        for i in range(n):
            x, y = positions[i]
            if self.agent_type == "rule":
                a = self.params.initial_alpha
                b = self.params.initial_beta
                if a < 0:
                    a = random.uniform(ALPHA_MIN, ALPHA_MAX)
                if b < 0:
                    b = random.uniform(BETA_MIN, BETA_MAX)
                self.agents.append(self._create_agent(x, y, random.uniform(1.0, 5.0), a, b))
            else:
                self.agents.append(self._create_agent(x, y, random.uniform(1.0, 5.0)))
        self._occupied = {(a.x, a.y) for a in self.agents}

    # ------------------------------------------------------------------
    # Perception
    # ------------------------------------------------------------------

    def get_percept(self, agent: AgentBase, occupied: Set[Tuple[int, int]]) -> List[float]:
        """Build the 10-dimensional percept, optionally with noisy neighbour food."""
        w, h = self.params.grid_width, self.params.grid_height
        grid = self.food_grid
        food_n: List[float] = []
        occ_n: List[float] = []
        for dx, dy in NEIGHBOR_OFFSETS:
            nx, ny = agent.x + dx, agent.y + dy
            if 0 <= nx < w and 0 <= ny < h:
                food_value = float(grid[nx, ny])
                if self.params.obs_noise_sigma > 0:
                    food_value = max(
                        0.0,
                        food_value + float(np.random.normal(0.0, self.params.obs_noise_sigma)),
                    )
                food_n.append(food_value)
                occ_n.append(1.0 if (nx, ny) in occupied else 0.0)
            else:
                food_n.append(-1.0)   # out-of-bounds food = -1
                occ_n.append(1.0)     # out-of-bounds treated as occupied
        return food_n + occ_n + [float(grid[agent.x, agent.y]), agent.energy]

    # ------------------------------------------------------------------
    # Sub-steps
    # ------------------------------------------------------------------

    def _regen_food(self) -> None:
        p = self.params
        self.food_grid += p.food_regen
        np.minimum(self.food_grid, p.max_food_cell, out=self.food_grid)
        self.total_food = float(self.food_grid.sum())

    def _move_agent(
        self, agent: AgentBase, dx: int, dy: int, occupied: Set[Tuple[int, int]]
    ) -> bool:
        if (dx, dy) == (0, 0):
            return False
        p = self.params
        nx, ny = agent.x + dx, agent.y + dy
        if not (0 <= nx < p.grid_width and 0 <= ny < p.grid_height):
            return False
        if (nx, ny) in occupied:
            return False
        occupied.discard((agent.x, agent.y))
        agent.x, agent.y = nx, ny
        agent.last_dir = (dx, dy)
        occupied.add((nx, ny))
        self.moves_step += 1
        return True

    def _handle_transfer(self, agent: AgentBase, x: int, y: int, energy_pre: float) -> None:
        """Transfer ft fraction of pre-move energy to the vacated cell."""
        cost = energy_pre * self.params.transfer_fraction
        agent.energy -= cost
        self.food_grid[x, y] += cost
        self.total_food += cost

    def _feed_agent(self, agent: AgentBase) -> float:
        p = self.params
        bite = min(p.max_bite, float(self.food_grid[agent.x, agent.y]))
        self.food_grid[agent.x, agent.y] -= bite
        self.total_food -= bite
        self.harvested_food_step += bite
        agent.energy += bite - p.base_metabolism
        return bite

    def _attempt_reproduction(
        self,
        agent: AgentBase,
        occupied: Set[Tuple[int, int]],
        should_reproduce: bool,
    ) -> Optional[AgentBase]:
        p = self.params
        if not should_reproduce:
            return None

        ax, ay = agent.x, agent.y
        sx, sy = -1, -1
        free_count = 0
        for dx, dy in NEIGHBOR_OFFSETS:
            nx, ny = ax + dx, ay + dy
            if 0 <= nx < p.grid_width and 0 <= ny < p.grid_height and (nx, ny) not in occupied:
                free_count += 1
                # Reservoir sampling keeps each free cell equally likely.
                if random.randint(1, free_count) == 1:
                    sx, sy = nx, ny

        if free_count == 0:
            return None

        agent.energy /= 2.0
        offspring_energy = agent.energy

        if isinstance(agent, NeuralAgent):
            offspring = self._create_agent(sx, sy, offspring_energy)
            if not self.random_offspring_net:
                offspring.net.copy_from(  # type: ignore[attr-defined]
                    agent.net,
                    mutate=self.mutate_offspring_net,
                    mutation_rate=p.net_mutation_rate,
                    mutation_strength=p.net_mutation_strength,
                )
        elif isinstance(agent, (RandomAgent, LLMAgent)):
            offspring = self._create_agent(sx, sy, offspring_energy)
        else:
            if self.random_offspring_propensity:
                new_alpha = random.uniform(ALPHA_MIN, ALPHA_MAX)
                new_beta = random.uniform(BETA_MIN, BETA_MAX)
            elif self.mutate_propensity:
                new_alpha = max(
                    ALPHA_MIN,
                    min(ALPHA_MAX,
                        agent.alpha + random.gauss(0, p.alpha_sigma)),  # type: ignore[attr-defined]
                )
                new_beta = max(
                    BETA_MIN,
                    min(BETA_MAX,
                        agent.beta + random.gauss(0, p.beta_sigma)),  # type: ignore[attr-defined]
                )
            else:
                new_alpha = agent.alpha  # type: ignore[attr-defined]
                new_beta = agent.beta  # type: ignore[attr-defined]
            offspring = self._create_agent(sx, sy, offspring_energy, new_alpha, new_beta)

        occupied.add((sx, sy))
        return offspring

    def _local_rewards(
        self,
        harvested_food: float,
        movement_cost: float,
        had_offspring: bool,
        survived: bool = True,
    ) -> Tuple[float, float]:
        """Return credit-separated movement and reproduction rewards.

        r_energy measures the next-harvest energetic consequence of movement:
            harvested food - metabolism - movement cost.
        Reproduction energy transfer is deliberately excluded so the movement
        policy is not blamed for the reproduction policy's action.

        r_fitness rewards successful births and reaching the next decision
        point. A terminal transition receives no survival bonus, but retains a
        birth bonus when reproduction at the preceding action actually
        succeeded.
        """
        p = self.params
        net_energy = float(harvested_food) - p.base_metabolism - float(movement_cost)
        r_energy = float(np.tanh(net_energy / max(p.rl_energy_scale, 1e-6)))
        r_energy = max(-p.rl_reward_clip, min(p.rl_reward_clip, r_energy))

        r_fitness = p.rl_survival_bonus if survived else 0.0
        if had_offspring:
            r_fitness += p.rl_birth_bonus
        r_fitness = max(-p.rl_reward_clip, min(p.rl_reward_clip, r_fitness))

        return r_energy, r_fitness

    def _apply_online_update(
        self,
        agent: AgentBase,
        r_energy: float,
        r_fitness: float,
        moved: bool,
        should_reproduce: bool = False,
        valid_decision: bool = True,
        can_reproduce: bool = True,
        food_ratio: float = 1.0,
        next_percept: Optional[List[float]] = None,
        done: bool = False,
    ) -> None:
        if not getattr(agent, "online_learning", False):
            return
        if isinstance(agent, RuleBasedAgent):
            agent.update_online(
                moved, r_energy, r_fitness,
                should_reproduce=should_reproduce,
                valid_decision=valid_decision,
                can_reproduce=can_reproduce,
                food_ratio=food_ratio,
            )
        elif hasattr(agent, "update_online"):
            agent.update_online(  # type: ignore[attr-defined]
                r_energy, r_fitness, next_percept=next_percept, done=done
            )

    def _finalize_pending_update(
        self,
        agent: AgentBase,
        harvested_food: float,
        next_percept: Optional[List[float]],
        done: bool,
    ) -> None:
        pending = getattr(agent, "_pending_update", None)
        if pending is None:
            return
        agent._pending_update = None  # type: ignore[attr-defined]
        r_energy, r_fitness = self._local_rewards(
            harvested_food=harvested_food,
            movement_cost=float(pending["movement_cost"]),
            had_offspring=bool(pending["had_offspring"]),
            survived=not done,
        )
        self.total_reward_step += r_energy + r_fitness
        self.reward_events_step += 1
        self._apply_online_update(
            agent,
            r_energy=r_energy,
            r_fitness=r_fitness,
            moved=bool(pending["moved"]),
            should_reproduce=bool(pending.get("should_reproduce", False)),
            valid_decision=bool(pending.get("valid_decision", True)),
            can_reproduce=bool(pending.get("can_reproduce", True)),
            food_ratio=float(pending.get("food_ratio", 1.0)),
            next_percept=next_percept,
            done=done,
        )

    # ------------------------------------------------------------------
    # Phase helpers
    # ------------------------------------------------------------------

    def _nutrition_phase(
        self, agents: List[AgentBase], occupied: Set[Tuple[int, int]]
    ) -> Tuple[List[AgentBase], List[List[float]]]:
        active: List[AgentBase] = []
        harvested: List[float] = []
        for agent in agents:
            harvested.append(self._feed_agent(agent))
            if agent.energy > 0:
                active.append(agent)
            else:
                occupied.discard((agent.x, agent.y))
                self.deaths_step += 1

        # Build each next state once, after all deaths have been removed. The
        # same percept is used for bootstrapping and for the upcoming action.
        percepts = [self.get_percept(a, occupied) for a in active]
        next_by_id = {id(a): p for a, p in zip(active, percepts)}
        for agent, bite in zip(agents, harvested):
            survived = agent.energy > 0
            self._finalize_pending_update(
                agent,
                harvested_food=bite,
                next_percept=next_by_id.get(id(agent)),
                done=not survived,
            )
        return active, percepts

    def _action_phase(
        self,
        agents: List[AgentBase],
        percepts: List[List[float]],
        occupied: Set[Tuple[int, int]],
    ) -> List[AgentBase]:
        actions = self._decide_actions_batch(agents, percepts)
        self.decision_records_step = [
            (
                agent,
                list(percept),
                (int(action[0]), int(action[1]), bool(action[2])),
            )
            for agent, percept, action in zip(agents, percepts, actions)
        ]
        next_gen: List[AgentBase] = []
        transfer_fraction = self.params.transfer_fraction
        action_map = {(0, -1): 0, (0, 1): 1, (1, 0): 2, (-1, 0): 3, (0, 0): 4}

        for agent, action in zip(agents, actions):
            energy_pre = agent.energy
            x_pre, y_pre = agent.x, agent.y
            dx, dy, should_reproduce = action

            # Track action counts for entropy/confidence metrics
            action_idx = action_map.get((dx, dy), 4)
            self.action_counts_step[action_idx] += 1
            self.total_agents_step += 1

            did_move = self._move_agent(agent, dx, dy, occupied)
            movement_cost = energy_pre * transfer_fraction if did_move else 0.0

            if did_move and transfer_fraction > 0:
                self._handle_transfer(agent, x_pre, y_pre, energy_pre)

            offspring = self._attempt_reproduction(agent, occupied, should_reproduce)
            if offspring is not None:
                next_gen.append(offspring)
                self.births_step += 1
                self.parent_ids_step.append(id(agent))

            survived = agent.energy > 0
            valid_decision = getattr(agent, "_last_valid_decision", True)

            can_reproduce = getattr(agent, "_last_can_reproduce", True)
            food_ratio = getattr(agent, "_last_food_ratio", 1.0)

            # Reward is part of the learning signal, not a general ecological
            # metric. Avoid creating and finalizing reward transitions for
            # variants whose policy is fixed during the episode.
            if getattr(agent, "online_learning", False):
                agent._pending_update = {  # type: ignore[attr-defined]
                    "movement_cost": movement_cost,
                    "had_offspring": (offspring is not None),
                    "moved": did_move,
                    "should_reproduce": bool(should_reproduce),
                    "valid_decision": valid_decision,
                    "can_reproduce": can_reproduce,
                    "food_ratio": food_ratio,
                }

            if not survived:
                self._finalize_pending_update(
                    agent, harvested_food=0.0, next_percept=None, done=True
                )

            if survived:
                agent.age += 1
                next_gen.append(agent)
            else:
                occupied.discard((agent.x, agent.y))
                self.deaths_step += 1

        return next_gen

    # ------------------------------------------------------------------
    # Main step
    # ------------------------------------------------------------------

    def step(self) -> None:
        self.moves_step = 0
        self.births_step = 0
        self.deaths_step = 0
        self.parent_ids_step = []
        self.harvested_food_step = 0.0
        self.action_counts_step = [0, 0, 0, 0, 0]
        self.total_reward_step = 0.0
        self.reward_events_step = 0
        self.total_agents_step = 0
        self.decision_records_step = []

        self._regen_food()
        random.shuffle(self.agents)
        occupied = self._occupied

        active, percepts = self._nutrition_phase(self.agents, occupied)
        self.agents = self._action_phase(active, percepts, occupied)
        self._occupied = occupied

    def _decide_actions_batch(
        self,
        agents: List[AgentBase],
        percepts: List[List[float]],
    ) -> List[Tuple[int, int, bool]]:
        if not agents:
            return []

        use_batch = (
            self.agent_type in MLP_OFFLINE_INIT
            and not self.net_learning
            and not self.mutate_offspring_net
            and self.net_template is not None
        )

        if use_batch:
            inp = np.array(percepts, dtype=np.float32)
            logits = self.net_template.forward_numpy(inp)
            logits = mask_logits_invalid_numpy(logits, inp)
            move_idx = np.argmax(logits[:, :5], axis=1).tolist()
            repro = (logits[:, 5] > 0).tolist()
            return [(ACTIONS[i][0], ACTIONS[i][1], bool(r)) for i, r in zip(move_idx, repro)]

        if self.agent_type == "llm":
            max_workers = LLMAgent.client().decision_max_workers
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                return list(pool.map(
                    lambda args: args[0].decide_action(args[1], self.params),
                    zip(agents, percepts),
                ))
        return [a.decide_action(percepts[i], self.params) for i, a in enumerate(agents)]

    def flush_online_learning(self) -> None:
        """Flush complete A2C buffers when an episode ends at a time limit."""
        for agent in self.agents:
            flush = getattr(agent, "flush_online", None)
            if callable(flush):
                flush()
