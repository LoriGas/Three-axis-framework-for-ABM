"""Massive parallel benchmark across scenarios, agents, and layer counts.

Generates:
- `results/analysis_full_layers.csv`
- `results/analysis_timeseries.csv`
- `results/analysis_replicates.csv`
"""

# NOTE: Behavioural metrics are tracked for every variant. Reward is tracked
# only for variants whose policy uses online learning.

from __future__ import annotations

import copy
import csv
import math
import multiprocessing as mp
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from config import (
    MLP_OFFLINE_INIT,
    MLP_VARIANT_TAGS,
    RESULTS_DIR,
    RULE_VARIANT_TAGS,
    SCENARIOS,
    VALID_SCENARIOS,
    WEIGHTS_DIR,
    get_params_for_scenario,
    setup_logging,
)
from model import World
from network import NeuralNetwork

console = Console()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

NUM_EPISODES: int = 300
MAX_STEPS: int = 2000
LAYERS_TO_TEST: List[int] = [1, 2, 3]
MAX_WORKERS: int = 10

ACTION_NAMES: List[str] = ["N", "S", "E", "W", "Stay"]

FULL_CSV: Path = RESULTS_DIR / "analysis_full_layers.csv"
TS_CSV: Path = RESULTS_DIR / "analysis_timeseries.csv"
REPLICATE_CSV: Path = RESULTS_DIR / "analysis_replicates.csv"

FULL_HEADER: List[str] = [
    "scenario", "layer", "mode", "avg_final_global",
    "avg_final_survived", "avg_last50_global", "avg_last50_survived",
    "avg_max_pop", "avg_steps_alive", "total_moves", "std_final_global",
    "std_final_survived", "std_last50_global", "std_last50_survived",
    "std_moves", "n_extinctions", "extinction_rate",
    "avg_movement_frequency", "std_movement_frequency",
    "avg_harvested_food", "std_harvested_food",
    "avg_reproduction_events", "std_reproduction_events",
    "avg_action_entropy", "std_action_entropy", "avg_policy_confidence",
    "std_policy_confidence", "avg_reward", "std_reward",
]

TS_HEADER: List[str] = [
    "scenario", "layer", "mode", "step", "avg_pop_step",
    "avg_prop_step", "std_prop_step", "avg_beta_step", "std_beta_step",
    "n_pop_samples", "n_prop_samples", "n_beta_samples",
    "avg_move_freq_step", "avg_harvested_food_step", "avg_births_step",
    "avg_entropy_step", "avg_confidence_step", "avg_reward_step",
]

REPLICATE_HEADER: List[str] = [
    "scenario", "layer", "mode", "replica", "seed", "pop_final",
    "avg_pop_last50", "pop_max", "extinct", "n_steps_alive",
    "total_moves", "movement_frequency", "avg_harvested_food",
    "reproduction_events", "action_entropy", "policy_confidence",
    "avg_reward",
]


@dataclass
class TestResult:
    """Outcome of a single episode."""

    pop_final: int
    avg_pop_last50: float
    pop_max: int
    extinct: int
    total_moves: int
    avg_prob_dist: List[float]
    pop_history: List[float]
    prop_history: List[float]
    beta_history: List[float]
    n_steps_alive: int
    # --- new behavioural metrics ---
    movement_frequency: float
    avg_harvested_food: float
    reproduction_events: int
    action_entropy: float
    policy_confidence: float
    avg_reward: float
    # per-step histories for timeseries
    move_freq_history: List[float]
    harvested_food_history: List[float]
    births_history: List[int]
    entropy_history: List[float]
    confidence_history: List[float]
    reward_history: List[float]


def _load_net(scenario: str, n_layers: int, weight_tag: str) -> Optional[NeuralNetwork]:
    weight_tag = weight_tag.strip().lower()
    weights_file = WEIGHTS_DIR / f"weights_{scenario}_L{n_layers}_{weight_tag}.pth"
    if not weights_file.exists():
        return None
    net = NeuralNetwork(num_hidden_layers=n_layers)
    net.load_state_dict(torch.load(weights_file, weights_only=True))
    net.eval()
    return net


def _run_episode(
    params,
    n_layers: int,
    *,
    mode: str,
    net_template: Optional[NeuralNetwork] = None,
    fixed_alpha: Optional[float] = None,
    fixed_beta: Optional[float] = None,
    mutate_propensity: bool = True,
    random_offspring_propensity: bool = False,
    propensity_learning: bool = False,
    net_learning: bool = False,
    seed: Optional[int] = None,
) -> TestResult:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    p = copy.copy(params)
    if fixed_alpha is not None:
        p.initial_alpha = float(fixed_alpha)
    if fixed_beta is not None:
        p.initial_beta = float(fixed_beta)

    world = World(
        p,
        agent_type=mode,
        net_template=net_template,
        mutate_propensity=mutate_propensity,
        random_offspring_propensity=random_offspring_propensity,
        propensity_learning=propensity_learning,
        net_learning=net_learning,
    )
    measure_reward = world.propensity_learning or world.net_learning

    pop_history: List[int] = []
    prop_history: List[float] = []
    beta_history: List[float] = []
    total_moves = 0
    n_steps_alive = MAX_STEPS

    # --- accumulators for new metrics ---
    total_agents_cumul = 0
    total_harvested = 0.0
    total_births = 0
    total_action_counts = [0, 0, 0, 0, 0]
    total_reward_cumul = 0.0
    total_reward_events_cumul = 0
    # per-step histories
    move_freq_history: List[float] = []
    harvested_food_history: List[float] = []
    births_history: List[int] = []
    entropy_history: List[float] = []
    confidence_history: List[float] = []
    reward_history: List[float] = []

    for step_i in range(MAX_STEPS):
        world.step()
        n = len(world.agents)
        pop_history.append(n)
        total_moves += world.moves_step

        # Collect new per-step metrics from World
        step_agents = world.total_agents_step
        total_agents_cumul += step_agents
        total_harvested += world.harvested_food_step
        total_births += world.births_step
        if measure_reward:
            total_reward_cumul += world.total_reward_step
            total_reward_events_cumul += world.reward_events_step
        for i in range(5):
            total_action_counts[i] += world.action_counts_step[i]

        # Per-step movement frequency
        mf = world.moves_step / step_agents if step_agents > 0 else 0.0
        move_freq_history.append(mf)

        # Per-step avg harvested food
        hf = world.harvested_food_step / step_agents if step_agents > 0 else 0.0
        harvested_food_history.append(hf)

        # Per-step births
        births_history.append(world.births_step)

        # Per-step action entropy and confidence
        if step_agents > 0:
            counts = np.array(world.action_counts_step, dtype=np.float64)
            probs = counts / counts.sum()
            probs_pos = probs[probs > 0]
            h = -float(np.sum(probs_pos * np.log(probs_pos)))
            max_h = math.log(5)
            conf = 1.0 - h / max_h if max_h > 0 else 1.0
        else:
            h = 0.0
            conf = 1.0
        entropy_history.append(h)
        confidence_history.append(conf)

        # Per-step avg reward
        rw = (
            world.total_reward_step / world.reward_events_step
            if world.reward_events_step > 0
            else (0.0 if measure_reward else float("nan"))
        )
        reward_history.append(rw)

        if mode == "rule" and n > 0:
            alive_agents = [a for a in world.agents if getattr(a, "alive", False)]
            propensities = [a.alpha for a in alive_agents if hasattr(a, "alpha")]
            betas = [a.beta for a in alive_agents if hasattr(a, "beta")]
            prop_history.append(float(np.mean(propensities)) if propensities else float("nan"))
            beta_history.append(float(np.mean(betas)) if betas else float("nan"))
        else:
            prop_history.append(float("nan"))
            beta_history.append(float("nan"))

        if n == 0:
            n_steps_alive = step_i + 1
            pop_history.extend([0] * (MAX_STEPS - len(pop_history)))
            prop_history.extend([float("nan")] * (MAX_STEPS - len(prop_history)))
            beta_history.extend([float("nan")] * (MAX_STEPS - len(beta_history)))
            move_freq_history.extend([0.0] * (MAX_STEPS - len(move_freq_history)))
            harvested_food_history.extend([0.0] * (MAX_STEPS - len(harvested_food_history)))
            births_history.extend([0] * (MAX_STEPS - len(births_history)))
            entropy_history.extend([0.0] * (MAX_STEPS - len(entropy_history)))
            confidence_history.extend([1.0] * (MAX_STEPS - len(confidence_history)))
            reward_history.extend([float("nan")] * (MAX_STEPS - len(reward_history)))
            break

    world.flush_online_learning()
    pop_final = int(pop_history[-1]) if pop_history else 0
    pop_max = int(max(pop_history)) if pop_history else 0
    extinct = 1 if pop_final == 0 else 0
    avg_prob_dist = [float("nan")] * len(ACTION_NAMES)

    avg_last50 = float(np.mean(pop_history[-50:])) if len(pop_history) >= 50 else float(np.mean(pop_history))

    # Aggregate new metrics over episode
    movement_frequency = total_moves / total_agents_cumul if total_agents_cumul > 0 else 0.0
    avg_harvested_food = total_harvested / total_agents_cumul if total_agents_cumul > 0 else 0.0
    if sum(total_action_counts) > 0:
        ac = np.array(total_action_counts, dtype=np.float64)
        probs_ep = ac / ac.sum()
        probs_ep_pos = probs_ep[probs_ep > 0]
        action_entropy = -float(np.sum(probs_ep_pos * np.log(probs_ep_pos)))
        policy_confidence = 1.0 - action_entropy / math.log(5) if math.log(5) > 0 else 1.0
    else:
        action_entropy = 0.0
        policy_confidence = 1.0
    avg_reward = (
        total_reward_cumul / total_reward_events_cumul
        if measure_reward and total_reward_events_cumul > 0
        else float("nan")
    )

    return TestResult(
        pop_final=pop_final,
        avg_pop_last50=avg_last50,
        pop_max=pop_max,
        extinct=extinct,
        total_moves=total_moves,
        avg_prob_dist=avg_prob_dist,
        pop_history=[float(v) for v in pop_history],
        prop_history=[float(v) for v in prop_history],
        beta_history=[float(v) for v in beta_history],
        n_steps_alive=n_steps_alive,
        movement_frequency=movement_frequency,
        avg_harvested_food=avg_harvested_food,
        reproduction_events=total_births,
        action_entropy=action_entropy,
        policy_confidence=policy_confidence,
        avg_reward=avg_reward,
        move_freq_history=move_freq_history,
        harvested_food_history=harvested_food_history,
        births_history=births_history,
        entropy_history=entropy_history,
        confidence_history=confidence_history,
        reward_history=reward_history,
    )


def _worker_benchmark(args: tuple) -> Optional[Tuple[List, List[List], List[List]]]:
    torch.set_num_threads(1)

    (
        scenario,
        layers,
        label,
        weight_tag,
        mode,
        fixed_alpha,
        fixed_beta,
        mutate_propensity,
        random_offspring_propensity,
        propensity_learning,
        net_learning,
        use_offline_weights,
    ) = args

    params = get_params_for_scenario(scenario)
    params.net_hidden_layers = layers

    net_template = None
    if use_offline_weights:
        net_template = _load_net(scenario, layers, weight_tag)
        if net_template is None:
            return None

    final_list_global: List[int] = []
    final_list_survived: List[int] = []
    last50_list_global: List[float] = []
    last50_list_survived: List[float] = []
    max_pop_list: List[int] = []
    steps_alive_list: List[int] = []
    ext_list: List[int] = []
    move_list: List[int] = []
    pop_sum_by_step = np.zeros(MAX_STEPS, dtype=np.float64)
    pop_count_by_step = np.zeros(MAX_STEPS, dtype=np.int32)
    prop_sum_by_step = np.zeros(MAX_STEPS, dtype=np.float64)
    prop_sq_sum_by_step = np.zeros(MAX_STEPS, dtype=np.float64)
    prop_count_by_step = np.zeros(MAX_STEPS, dtype=np.int32)
    beta_sum_by_step = np.zeros(MAX_STEPS, dtype=np.float64)
    beta_sq_sum_by_step = np.zeros(MAX_STEPS, dtype=np.float64)
    beta_count_by_step = np.zeros(MAX_STEPS, dtype=np.int32)
    # --- new metrics accumulators ---
    mf_list: List[float] = []
    hf_list: List[float] = []
    repro_list: List[int] = []
    entropy_list: List[float] = []
    confidence_list: List[float] = []
    reward_list: List[float] = []
    replicate_rows: List[List] = []
    # timeseries accumulators for new metrics
    mf_sum_by_step = np.zeros(MAX_STEPS, dtype=np.float64)
    hf_sum_by_step = np.zeros(MAX_STEPS, dtype=np.float64)
    births_sum_by_step = np.zeros(MAX_STEPS, dtype=np.float64)
    ent_sum_by_step = np.zeros(MAX_STEPS, dtype=np.float64)
    conf_sum_by_step = np.zeros(MAX_STEPS, dtype=np.float64)
    rew_sum_by_step = np.zeros(MAX_STEPS, dtype=np.float64)

    scenario_offset = VALID_SCENARIOS.index(scenario) * 100_000
    layer_offset = int(layers) * 10_000
    for replica in range(NUM_EPISODES):
        # The same scenario/layer/replica seed is reused across variants. This
        # preserves independent episodes and supports paired follow-up analyses.
        episode_seed = 1_000_000 + scenario_offset + layer_offset + replica
        res = _run_episode(
            params,
            layers,
            mode=mode,
            net_template=net_template,
            fixed_alpha=fixed_alpha,
            fixed_beta=fixed_beta,
            mutate_propensity=mutate_propensity,
            random_offspring_propensity=random_offspring_propensity,
            propensity_learning=propensity_learning,
            net_learning=net_learning,
            seed=episode_seed,
        )
        final_list_global.append(res.pop_final)
        last50_list_global.append(res.avg_pop_last50)
        max_pop_list.append(res.pop_max)
        steps_alive_list.append(res.n_steps_alive)
        ext_list.append(res.extinct)
        move_list.append(res.total_moves)

        # new metrics
        mf_list.append(res.movement_frequency)
        hf_list.append(res.avg_harvested_food)
        repro_list.append(res.reproduction_events)
        entropy_list.append(res.action_entropy)
        confidence_list.append(res.policy_confidence)
        reward_list.append(res.avg_reward)
        replicate_rows.append([
            scenario,
            layers,
            label,
            replica,
            episode_seed,
            res.pop_final,
            res.avg_pop_last50,
            res.pop_max,
            res.extinct,
            res.n_steps_alive,
            res.total_moves,
            res.movement_frequency,
            res.avg_harvested_food,
            res.reproduction_events,
            res.action_entropy,
            res.policy_confidence,
            res.avg_reward,
        ])

        if res.extinct == 0:
            final_list_survived.append(res.pop_final)
            last50_list_survived.append(res.avg_pop_last50)

        for t in range(res.n_steps_alive):
            pop_sum_by_step[t] += float(res.pop_history[t])
            pop_count_by_step[t] += 1
            mf_sum_by_step[t] += float(res.move_freq_history[t])
            hf_sum_by_step[t] += float(res.harvested_food_history[t])
            births_sum_by_step[t] += float(res.births_history[t])
            ent_sum_by_step[t] += float(res.entropy_history[t])
            conf_sum_by_step[t] += float(res.confidence_history[t])
            rew_sum_by_step[t] += float(res.reward_history[t])
        for t in range(res.n_steps_alive):
            value = res.prop_history[t]
            if not np.isnan(value):
                prop_sum_by_step[t] += float(value)
                prop_sq_sum_by_step[t] += float(value) ** 2
                prop_count_by_step[t] += 1
            bval = res.beta_history[t]
            if not np.isnan(bval):
                beta_sum_by_step[t] += float(bval)
                beta_sq_sum_by_step[t] += float(bval) ** 2
                beta_count_by_step[t] += 1

    avg_final_global = float(np.mean(final_list_global))
    std_final_global = float(np.std(final_list_global, ddof=1)) if len(final_list_global) > 1 else float("nan")
    avg_final_survived = float(np.mean(final_list_survived)) if final_list_survived else float("nan")
    std_final_survived = float(np.std(final_list_survived, ddof=1)) if len(final_list_survived) > 1 else float("nan")
    avg_last50_global = float(np.mean(last50_list_global))
    std_last50_global = float(np.std(last50_list_global, ddof=1)) if len(last50_list_global) > 1 else float("nan")
    avg_last50_survived = float(np.mean(last50_list_survived)) if last50_list_survived else float("nan")
    std_last50_survived = float(np.std(last50_list_survived, ddof=1)) if len(last50_list_survived) > 1 else float("nan")
    avg_max_pop = float(np.mean(max_pop_list))
    avg_steps_alive = float(np.mean(steps_alive_list))
    avg_moves = float(np.mean(move_list))
    std_moves = float(np.std(move_list, ddof=1)) if len(move_list) > 1 else float("nan")
    extinction_rate = float(np.mean(ext_list)) if ext_list else float("nan")

    # Aggregate new metrics
    avg_mf = float(np.mean(mf_list)) if mf_list else float("nan")
    std_mf = float(np.std(mf_list, ddof=1)) if len(mf_list) > 1 else float("nan")
    avg_hf = float(np.mean(hf_list)) if hf_list else float("nan")
    std_hf = float(np.std(hf_list, ddof=1)) if len(hf_list) > 1 else float("nan")
    avg_repro = float(np.mean(repro_list)) if repro_list else float("nan")
    std_repro = float(np.std(repro_list, ddof=1)) if len(repro_list) > 1 else float("nan")
    avg_ent = float(np.mean(entropy_list)) if entropy_list else float("nan")
    std_ent = float(np.std(entropy_list, ddof=1)) if len(entropy_list) > 1 else float("nan")
    avg_conf = float(np.mean(confidence_list)) if confidence_list else float("nan")
    std_conf = float(np.std(confidence_list, ddof=1)) if len(confidence_list) > 1 else float("nan")
    avg_rew = float(np.mean(reward_list)) if reward_list else float("nan")
    std_rew = float(np.std(reward_list, ddof=1)) if len(reward_list) > 1 else float("nan")

    summary_row = [
        scenario,
        layers,
        label,
        avg_final_global,
        avg_final_survived,
        avg_last50_global,
        avg_last50_survived,
        avg_max_pop,
        avg_steps_alive,
        avg_moves,
        std_final_global,
        std_final_survived,
        std_last50_global,
        std_last50_survived,
        std_moves,
        int(np.sum(ext_list)),
        extinction_rate,
        avg_mf,
        std_mf,
        avg_hf,
        std_hf,
        avg_repro,
        std_repro,
        avg_ent,
        std_ent,
        avg_conf,
        std_conf,
        avg_rew,
        std_rew,
    ]

    ts_rows: List[List] = []
    for step in range(MAX_STEPS):
        if pop_count_by_step[step] == 0:
            continue
        nc = pop_count_by_step[step]
        avg_pop_step = float(pop_sum_by_step[step] / nc)
        if prop_count_by_step[step] > 0:
            n = prop_count_by_step[step]
            avg_prop_step = float(prop_sum_by_step[step] / n)
            variance = float(prop_sq_sum_by_step[step] / n) - avg_prop_step**2
            std_prop_step = float(np.sqrt(max(0.0, variance)))
        else:
            avg_prop_step = float("nan")
            std_prop_step = float("nan")
        if beta_count_by_step[step] > 0:
            nb = beta_count_by_step[step]
            avg_beta_step = float(beta_sum_by_step[step] / nb)
            variance_b = float(beta_sq_sum_by_step[step] / nb) - avg_beta_step**2
            std_beta_step = float(np.sqrt(max(0.0, variance_b)))
        else:
            avg_beta_step = float("nan")
            std_beta_step = float("nan")
        ts_rows.append(
            [
                scenario,
                layers,
                label,
                step,
                avg_pop_step,
                avg_prop_step,
                std_prop_step,
                avg_beta_step,
                std_beta_step,
                int(nc),
                int(prop_count_by_step[step]),
                int(beta_count_by_step[step]),
                float(mf_sum_by_step[step] / nc),
                float(hf_sum_by_step[step] / nc),
                float(births_sum_by_step[step] / nc),
                float(ent_sum_by_step[step] / nc),
                float(conf_sum_by_step[step] / nc),
                float(rew_sum_by_step[step] / nc),
            ]
        )

    return summary_row, ts_rows, replicate_rows


_EVO_TOP5_TAG = "sup_rb_evo_top5_auto"


def _build_tasks() -> List[tuple]:
    tasks: List[tuple] = []
    for scenario in VALID_SCENARIOS:
        tasks.append((scenario, 0, "Random", None, "random", None, None, False, False, False, False, False))
        tasks.append((scenario, 0, "Rule-Random/Fix", None, "rule", None, None, False, False, False, False, False))
        tasks.append((scenario, 0, "Rule-Random/Random", None, "rule", None, None, False, True, False, False, False))
        tasks.append((scenario, 0, "Rule-Random/Evo", None, "rule", None, None, True, False, False, False, False))
        tasks.append((scenario, 0, "Rule-Random/Learn", None, "rule", None, None, False, False, True, False, False))
        tasks.append((scenario, 0, "Rule-Fixed/Fix", None, "rule", 1.0, 0.5, False, False, False, False, False))
        tasks.append((scenario, 0, "Rule-Fixed/Evo", None, "rule", 1.0, 0.5, True, False, False, False, False))
        tasks.append((scenario, 0, "Rule-Fixed/Learn", None, "rule", 1.0, 0.5, False, False, True, False, False))

        for layers in LAYERS_TO_TEST:
            tasks.append((scenario, layers, "MLP-Random/Fix",             None,          "mlp_rand_fix",      None, None, True, False, False, False, False))
            tasks.append((scenario, layers, "MLP-Random/Random",          None,          "mlp_rand_rand",     None, None, True, False, False, False, False))
            tasks.append((scenario, layers, "MLP-Random/Evo",             None,          "mlp_rand_evo",      None, None, True, False, False, False, False))
            tasks.append((scenario, layers, "MLP-Random/Learn",           None,          "mlp_rand_learn",    None, None, True, False, False, True,  False))
            tasks.append((scenario, layers, "MLP-EvoTop5/Fix",            _EVO_TOP5_TAG, "mlp_offline_fix",   None, None, True, False, False, False, True))
            tasks.append((scenario, layers, "MLP-EvoTop5/Evo",            _EVO_TOP5_TAG, "mlp_offline_evo",   None, None, True, False, False, False, True))
            tasks.append((scenario, layers, "MLP-EvoTop5/Learn",          _EVO_TOP5_TAG, "mlp_offline_learn", None, None, True, False, False, True,  True))
    return tasks


def _row_key(row: List) -> Tuple[str, int, str]:
    return str(row[0]), int(float(row[1])), str(row[2])


def _load_csv_rows(path: Path) -> List[List[str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with open(path, newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        return [row for row in reader if row]


def _atomic_write_csv(path: Path, header: List[str], rows: List[List]) -> None:
    """Replace a CSV only after its complete temporary copy is on disk."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _save_results(
    summary_rows: List[List],
    ts_rows: List[List],
    replicate_rows: List[List],
) -> None:
    """Persist every completed condition, preserving deterministic ordering."""
    summary_rows.sort(key=lambda row: (row[0], int(float(row[1])), row[2]))
    ts_rows.sort(
        key=lambda row: (row[0], int(float(row[1])), row[2], int(float(row[3])))
    )
    replicate_rows.sort(
        key=lambda row: (row[0], int(float(row[1])), row[2], int(float(row[3])))
    )
    # Write the large time-series file first.  The compact summary therefore
    # never advertises a condition whose time series has not yet been saved.
    _atomic_write_csv(TS_CSV, TS_HEADER, ts_rows)
    _atomic_write_csv(REPLICATE_CSV, REPLICATE_HEADER, replicate_rows)
    _atomic_write_csv(FULL_CSV, FULL_HEADER, summary_rows)


def _parse_selection(raw: str, allowed: List[str], *, numeric: bool = False) -> List[str]:
    """Parse comma-separated values; blank or 'all' selects everything."""
    raw = raw.strip()
    if not raw or raw.lower() in {"all", "*"}:
        return list(allowed)
    requested = [part.strip() for part in raw.split(",") if part.strip()]
    if numeric:
        by_index = {str(i + 1): value for i, value in enumerate(allowed)}
        requested = [by_index.get(value, value) for value in requested]
    invalid = [value for value in requested if value not in allowed]
    if invalid:
        raise ValueError(", ".join(invalid))
    return list(dict.fromkeys(requested))


def _choose_tasks(all_tasks: List[tuple]) -> Tuple[List[tuple], bool]:
    """Choose a full rewrite or a targeted replacement of existing rows."""
    outputs_exist = FULL_CSV.exists() or TS_CSV.exists()
    if not outputs_exist:
        console.print("[dim]No previous results found; a new output will be created.[/dim]")
        return all_tasks, False

    while True:
        try:
            answer = input(
                "  Rewrite everything or rerun only selected scenarios/variants? [a/s] "
            ).strip().lower()
        except EOFError:
            console.print("[yellow]Input unavailable; rewriting everything.[/yellow]")
            return all_tasks, False
        if answer in {"a", "all", "rewrite"}:
            return all_tasks, False
        if answer in {"s", "select", "selected"}:
            break
        console.print("[yellow]Enter 'a' for all or 's' for a targeted selection.[/yellow]")

    scenarios = list(VALID_SCENARIOS)
    variants = list(dict.fromkeys(task[2] for task in all_tasks))
    layers = [str(layer) for layer in LAYERS_TO_TEST]

    console.print("[bold]Targeted selection[/bold] (press Enter or type 'all' to select all)")
    console.print("  Available scenarios: " + ", ".join(scenarios))
    console.print("  Available variants:")
    for idx, variant in enumerate(variants, 1):
        console.print(f"    {idx:>2}. {variant}")
    console.print("  Available MLP layers: " + ", ".join(f"L{x}" for x in layers))

    while True:
        try:
            selected_scenarios = _parse_selection(input("  Scenarios: "), scenarios)
            selected_variants = _parse_selection(
                input("  Variants (names or numbers): "), variants, numeric=True
            )
            raw_layers = input("  MLP layers: ").upper().replace("L", "")
            selected_layers = _parse_selection(raw_layers, layers)
            break
        except ValueError as exc:
            console.print(f"[yellow]Invalid values: {exc}. Try again.[/yellow]")
        except EOFError:
            console.print("[yellow]Input interrupted; using all combinations.[/yellow]")
            return all_tasks, False

    selected = [
        task for task in all_tasks
        if task[0] in selected_scenarios
        and task[2] in selected_variants
        and (task[1] == 0 or str(task[1]) in selected_layers)
    ]
    return selected, True


def main() -> None:
    t_start = time.perf_counter()

    console.print()
    console.print(
        Panel.fit(
            "[bold white]BENCHMARK ANALYSIS[/bold white]\n"
            "[dim]Scenarios x agents x layer counts[/dim]",
            border_style="bright_cyan",
            padding=(1, 4),
        )
    )

    cfg_table = Table(show_header=False, box=None, padding=(0, 2))
    cfg_table.add_column(style="bold cyan")
    cfg_table.add_column(style="white")
    cfg_table.add_row("Episodes per condition", str(NUM_EPISODES))
    cfg_table.add_row("Max steps per episode", str(MAX_STEPS))
    cfg_table.add_row("Layers tested", ", ".join(f"L{l}" for l in LAYERS_TO_TEST))
    cfg_table.add_row("Workers", str(MAX_WORKERS))
    console.print(Panel(cfg_table, title="[bold]Configuration[/bold]", border_style="dim", expand=False))

    all_tasks = _build_tasks()
    tasks, targeted_update = _choose_tasks(all_tasks)
    total = len(tasks)
    console.print()
    console.print(
        f"  [bold]Total conditions:[/bold] [yellow]{total}[/yellow]  "
        f"[dim]({total} x {NUM_EPISODES} = {total * NUM_EPISODES:,} episodes)[/dim]"
    )
    console.print()

    ctx = mp.get_context("spawn")
    summary_rows: List[List] = _load_csv_rows(FULL_CSV) if targeted_update else []
    ts_rows: List[List] = _load_csv_rows(TS_CSV) if targeted_update else []
    replicate_rows: List[List] = (
        _load_csv_rows(REPLICATE_CSV) if targeted_update else []
    )
    skipped = 0

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold]{task.description}"),
            BarColumn(bar_width=40, complete_style="green", finished_style="bright_green"),
            "[progress.percentage]{task.percentage:>3.0f}%",
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task_id = progress.add_task("Benchmark in progress...", total=total)

            with ctx.Pool(processes=MAX_WORKERS) as pool:
                for payload in pool.imap_unordered(_worker_benchmark, tasks):
                    if payload is None:
                        skipped += 1
                    else:
                        summary_row, partial_ts_rows, partial_replicate_rows = payload
                        completed_key = _row_key(summary_row)
                        if targeted_update:
                            summary_rows = [
                                row for row in summary_rows
                                if _row_key(row) != completed_key
                            ]
                            ts_rows = [
                                row for row in ts_rows
                                if _row_key(row) != completed_key
                            ]
                            replicate_rows = [
                                row for row in replicate_rows
                                if _row_key(row) != completed_key
                            ]
                        summary_rows.append(summary_row)
                        ts_rows.extend(partial_ts_rows)
                        replicate_rows.extend(partial_replicate_rows)
                        _save_results(summary_rows, ts_rows, replicate_rows)
                        scenario_name, layer_value, variant_name = completed_key
                        layer_label = (
                            f"L{layer_value}" if layer_value > 0 else "no layer"
                        )
                        progress.console.print(
                            "  [bold green]✓ Completed and saved[/bold green] "
                            f"{scenario_name} · {variant_name} · {layer_label}"
                        )
                    progress.advance(task_id)

            progress.update(task_id, description="[bold bright_green]Benchmark completed!")
    except KeyboardInterrupt:
        console.print(
            "\n[yellow]Analysis interrupted. Completed conditions were saved; "
            "only the condition currently running will be lost.[/yellow]"
        )
        return

    # Also normalise the outputs when every selected condition was skipped.
    _save_results(summary_rows, ts_rows, replicate_rows)

    elapsed = time.perf_counter() - t_start
    minutes, seconds = divmod(int(elapsed), 60)

    console.print()
    summary = Table.grid(padding=(0, 2))
    summary.add_column(style="bold")
    summary.add_column()
    summary.add_row("[green]Rows written[/green]", f"{len(summary_rows)}")
    summary.add_row("[green]Timeseries rows[/green]", f"{len(ts_rows)}")
    summary.add_row("[green]Episode rows[/green]", f"{len(replicate_rows)}")
    summary.add_row("[yellow]Skipped conditions[/yellow]", f"{skipped}")
    summary.add_row("[cyan]Total time[/cyan]", f"{minutes}m {seconds}s")
    summary.add_row("[blue]Output[/blue]", str(FULL_CSV))
    summary.add_row("[blue]TS output[/blue]", str(TS_CSV))
    summary.add_row("[blue]Episode output[/blue]", str(REPLICATE_CSV))

    console.print(
        Panel(
            summary,
            title="[bold bright_green]Benchmark completed[/bold bright_green]",
            border_style="bright_green",
            expand=False,
        )
    )
    console.print()


if __name__ == "__main__":
    # Required before normal application setup when Windows starts spawned
    # workers (and when the script is packaged as an executable).
    mp.freeze_support()
    setup_logging()
    main()
