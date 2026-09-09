"""Dedicated benchmark for the LLM agent.

Generates CSVs with the same schema as run_analysis.py:
- `results/analysis_llm_full_layers_<model>.csv`
- `results/analysis_llm_timeseries_<model>.csv`
- `results/analysis_llm_simulations_<model>.csv`
- `results/llm_decisions_<model>.csv`  (every individual decision)

By default this script forces the LLMAgent local fallback policy so accidental
API costs are avoided. To use a real OpenAI-compatible endpoint, set:

    LLM_ENABLED=1 LLM_API_KEY=... LLM_MODEL=...

For the same interactive workflow with GPT-5 mini and model-isolated filenames,
use `python run_gpt5_mini_benchmarks.py`.
"""

from __future__ import annotations

import copy
import csv
import json
import math
import multiprocessing as mp
import os
import random
import time
from dataclasses import dataclass, field
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
    ACTIONS,
    RESULTS_DIR,
    VALID_SCENARIOS,
    get_params_for_scenario,
    model_filename_slug,
    setup_logging,
)
from model import World

console = Console()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Temporary fixed setup requested: custom simulations per scenario, 400 steps each.
SCENARIO_EPISODES = {
    "S1": 1,
    "S2": 5,
    "S3": 5,
    "S4": 5,
    "S5": 1,
}
MAX_STEPS: int = 400
MAX_WORKERS: int = int(os.getenv("LLM_ANALYSIS_MAX_WORKERS", "1"))
REJECT_FAILED_EPISODES: bool = os.getenv(
    "LLM_REJECT_FAILED_EPISODES", "1"
).strip().lower() not in {"0", "false", "no"}
MAX_EPISODE_RESTARTS: int = max(
    0, int(os.getenv("LLM_EPISODE_MAX_RESTARTS", "5"))
)
MAX_STEP_RESUMES: int = max(
    0, int(os.getenv("LLM_STEP_MAX_RESUMES", "5"))
)

ACTION_NAMES: List[str] = ["N", "S", "E", "W", "Stay"]

LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"
LLM_MODEL_SLUG: str = model_filename_slug(LLM_MODEL)
PROMPT_VARIANTS = {"goal", "no_goal"}
PROMPT_VARIANT: str = "goal"

FULL_CSV: Path = RESULTS_DIR / f"analysis_llm_full_layers_{LLM_MODEL_SLUG}.csv"
TS_CSV: Path = RESULTS_DIR / f"analysis_llm_timeseries_{LLM_MODEL_SLUG}.csv"
SIM_CSV: Path = RESULTS_DIR / f"analysis_llm_simulations_{LLM_MODEL_SLUG}.csv"
DECISIONS_CSV: Path = RESULTS_DIR / f"llm_decisions_{LLM_MODEL_SLUG}.csv"
REJECTED_CSV: Path = RESULTS_DIR / f"analysis_llm_rejected_{LLM_MODEL_SLUG}.csv"


def _configure_prompt_variant(variant: str) -> None:
    """Select prompt treatment and isolate every output file for that treatment."""
    global PROMPT_VARIANT, FULL_CSV, TS_CSV, SIM_CSV, DECISIONS_CSV, REJECTED_CSV
    PROMPT_VARIANT = variant if variant in PROMPT_VARIANTS else "goal"
    suffix = "" if PROMPT_VARIANT == "goal" else "_no_goal"
    FULL_CSV = RESULTS_DIR / f"analysis_llm_full_layers{suffix}_{LLM_MODEL_SLUG}.csv"
    TS_CSV = RESULTS_DIR / f"analysis_llm_timeseries{suffix}_{LLM_MODEL_SLUG}.csv"
    SIM_CSV = RESULTS_DIR / f"analysis_llm_simulations{suffix}_{LLM_MODEL_SLUG}.csv"
    DECISIONS_CSV = RESULTS_DIR / f"llm_decisions{suffix}_{LLM_MODEL_SLUG}.csv"
    REJECTED_CSV = RESULTS_DIR / f"analysis_llm_rejected{suffix}_{LLM_MODEL_SLUG}.csv"
    os.environ["LLM_PROMPT_VARIANT"] = PROMPT_VARIANT
    os.environ["LLM_LOG_FILE"] = str(
        RESULTS_DIR / f"llm_calls{suffix}_{LLM_MODEL_SLUG}.jsonl"
    )


def _ask_prompt_variant() -> str:
    default = os.getenv("LLM_PROMPT_VARIANT", "goal").strip().lower()
    if default not in PROMPT_VARIANTS:
        default = "goal"
    if os.getenv("LLM_PROMPT_VARIANT_LOCKED", "0").strip().lower() in {"1", "true", "yes"}:
        console.print(f"[dim]Preselected system-prompt treatment: {default}[/dim]")
        return default
    console.print("\n  [bold]System-prompt treatment:[/bold]")
    console.print("    [cyan]1)[/cyan] goal     — current prompt: explicit survival/reproduction goal")
    console.print("    [cyan]2)[/cyan] no_goal  — neutral prompt: no explicit objective")
    try:
        choice = input(f"\n  Choose prompt [1/2, default {'2' if default == 'no_goal' else '1'}]: ").strip()
    except EOFError:
        choice = ""
    if not choice:
        return default
    return "no_goal" if choice == "2" else "goal"

SUMMARY_HEADER: List[str] = [
    "scenario",
    "layer",
    "mode",
    "avg_final_global",
    "avg_final_survived",
    "avg_last50_global",
    "avg_last50_survived",
    "avg_max_pop",
    "avg_steps_alive",
    "total_moves",
    "std_final_global",
    "std_final_survived",
    "std_last50_global",
    "std_last50_survived",
    "std_moves",
    "n_extinctions",
    "extinction_rate",
    "avg_movement_frequency",
    "std_movement_frequency",
    "avg_harvested_food",
    "std_harvested_food",
    "avg_reproduction_events",
    "std_reproduction_events",
    "avg_action_entropy",
    "std_action_entropy",
    "avg_policy_confidence",
    "std_policy_confidence",
]

TS_HEADER: List[str] = [
    "scenario",
    "layer",
    "mode",
    "step",
    "avg_pop_step",
    "avg_prop_step",
    "std_prop_step",
    "n_pop_samples",
    "n_prop_samples",
    "avg_move_freq_step",
    "avg_harvested_food_step",
    "avg_births_step",
    "avg_entropy_step",
    "avg_confidence_step",
]

SIM_HEADER: List[str] = [
    "scenario",
    "episode",
    "layer",
    "mode",
    "pop_final",
    "avg_pop_last50",
    "pop_max",
    "extinct",
    "total_moves",
    "n_steps_alive",
    "pop_history",
    "movement_frequency",
    "avg_harvested_food",
    "reproduction_events",
    "action_entropy",
    "policy_confidence",
    "technical_failures",
    "cache_hits",
    "uncached_decisions",
]

DECISIONS_HEADER: List[str] = [
    "scenario",
    "episode",
    "step",
    "agent_id",
    "food_N",
    "food_S",
    "food_E",
    "food_W",
    "occ_N",
    "occ_S",
    "occ_E",
    "occ_W",
    "food_here",
    "energy",
    "free_cells",
    "move",
    "reproduce",
    "did_reproduce",
    "did_move",
    "cached",
    "error",
]

REJECTED_HEADER: List[str] = [
    "scenario",
    "episode",
    "attempt",
    "steps_completed",
    "decisions_before_rejection",
    "technical_failures",
    "errors",
]


@dataclass
class DecisionRecord:
    """Single decision made by an LLM agent."""
    scenario: str
    episode: int
    step: int
    agent_id: int
    food_N: float
    food_S: float
    food_E: float
    food_W: float
    occ_N: int
    occ_S: int
    occ_E: int
    occ_W: int
    food_here: float
    energy: float
    free_cells: int
    move: str
    reproduce: bool
    did_reproduce: bool
    did_move: bool
    cached: bool
    error: Optional[str]


@dataclass
class TestResult:
    """Outcome of a single LLM episode."""

    pop_final: int
    avg_pop_last50: float
    pop_max: int
    extinct: int
    total_moves: int
    pop_history: List[float]
    n_steps_alive: int
    # --- new behavioural metrics ---
    movement_frequency: float
    avg_harvested_food: float
    reproduction_events: int
    action_entropy: float
    policy_confidence: float
    # per-step histories for timeseries
    move_freq_history: List[float] = field(default_factory=list)
    harvested_food_history: List[float] = field(default_factory=list)
    births_history: List[int] = field(default_factory=list)
    entropy_history: List[float] = field(default_factory=list)
    confidence_history: List[float] = field(default_factory=list)
    # decision records for this episode
    decisions: List[DecisionRecord] = field(default_factory=list)
    technical_failures: int = 0
    cache_hits: int = 0
    uncached_decisions: int = 0
    errors: List[str] = field(default_factory=list)


@dataclass
class ExistingRunState:
    """Previously persisted LLM simulation rows and time-series aggregates."""

    results_by_scenario: Dict[str, List[TestResult]]
    next_episode_by_scenario: Dict[str, int]
    ts_pop_sum_by_scenario: Dict[str, np.ndarray]
    ts_pop_count_by_scenario: Dict[str, np.ndarray]


@dataclass
class RejectedEpisodeAttempt:
    """Compact diagnostic record for one automatically discarded attempt."""

    attempt: int
    steps_completed: int
    decisions_before_rejection: int
    technical_failures: int
    errors: List[str]


def _run_episode(
    params,
    scenario: str = "",
    episode_idx: int = 0,
    attempt: int = 0,
) -> TestResult:
    scenario_offset = VALID_SCENARIOS.index(scenario) if scenario in VALID_SCENARIOS else 0
    # An automatic restart repeats the same experimental replication.  Keep
    # its simulation seed independent of the technical attempt number so a
    # network fault does not silently change the stochastic conditions.
    seed = 9_000_000 + scenario_offset * 100_000 + episode_idx * 100
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    p = copy.copy(params)
    world = World(
        p,
        agent_type="llm",
        mutate_propensity=False,
        random_offspring_propensity=False,
        propensity_learning=False,
        net_learning=False,
    )

    pop_history: List[int] = []
    total_moves = 0
    n_steps_alive = MAX_STEPS

    # --- accumulators for new metrics ---
    total_agents_cumul = 0
    total_harvested = 0.0
    total_births = 0
    total_action_counts = [0, 0, 0, 0, 0]
    move_freq_history: List[float] = []
    harvested_food_history: List[float] = []
    births_history: List[int] = []
    entropy_history: List[float] = []
    confidence_history: List[float] = []

    # --- decision logging ---
    decisions: List[DecisionRecord] = []
    action_map = {(0, -1): 0, (0, 1): 1, (1, 0): 2, (-1, 0): 3, (0, 0): 4}
    technical_failures = 0
    cache_hits = 0
    uncached_decisions = 0
    errors: List[str] = []

    step_i = 0
    step_resume_attempt = 0
    while step_i < MAX_STEPS:
        # Keep the last fully valid simulation state.  If an API request fails
        # during this step, discard every mutation made by the incomplete step
        # and repeat it from this checkpoint with the same random-generator
        # states.  Successfully parsed LLM responses may remain in the client
        # cache; failed fallback decisions are never cached.
        world_checkpoint = copy.deepcopy(world)
        python_random_checkpoint = random.getstate()
        numpy_random_checkpoint = np.random.get_state()
        torch_random_checkpoint = torch.get_rng_state()

        # Positions are captured only to measure whether an executed policy
        # request resulted in movement. Percepts and actions come exclusively
        # from World.decision_records_step after the real nutrition phase.
        positions_pre = {id(a): (a.x, a.y) for a in world.agents}

        world.step()

        step_errors = [
            str(error)
            for agent, _, _ in world.decision_records_step
            if (error := getattr(agent, "last_decision_error", None))
        ]
        if step_errors and REJECT_FAILED_EPISODES:
            if step_resume_attempt < MAX_STEP_RESUMES:
                step_resume_attempt += 1
                world = world_checkpoint
                random.setstate(python_random_checkpoint)
                np.random.set_state(numpy_random_checkpoint)
                torch.set_rng_state(torch_random_checkpoint)
                print(
                    "[STEP-RESUME] "
                    f"{scenario} · episode {episode_idx} · step {step_i} · "
                    f"retry {step_resume_attempt}/{MAX_STEP_RESUMES}; "
                    "resuming from the last valid step.",
                    flush=True,
                )
                continue

            technical_failures = len(step_errors)
            errors.extend(step_errors)
            n_steps_alive = step_i
            break

        step_resume_attempt = 0

        n = len(world.agents)
        pop_history.append(n)
        total_moves += world.moves_step

        # Collect new per-step metrics
        step_agents = world.total_agents_step
        total_agents_cumul += step_agents
        total_harvested += world.harvested_food_step
        total_births += world.births_step
        for i in range(5):
            total_action_counts[i] += world.action_counts_step[i]

        mf = world.moves_step / step_agents if step_agents > 0 else 0.0
        move_freq_history.append(mf)
        hf = world.harvested_food_step / step_agents if step_agents > 0 else 0.0
        harvested_food_history.append(hf)
        births_history.append(world.births_step)

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

        # --- Log only decisions actually requested during the action phase ---
        parent_ids_set = set(world.parent_ids_step)
        for agent, percept, action in world.decision_records_step:
            aid = id(agent)
            pos_pre = positions_pre.get(aid, (agent.x, agent.y))
            did_move = (agent.x, agent.y) != pos_pre
            action_idx = action_map.get((int(action[0]), int(action[1])), 4)
            move_name = ACTION_NAMES[action_idx]
            did_reproduce = aid in parent_ids_set
            occ = percept[4:8]
            free_cells = sum(1 for o in occ if o < 0.5)
            reproduce_intended = bool(action[2])
            error = getattr(agent, "last_decision_error", None)
            cached = bool(getattr(agent, "last_decision_cached", False))
            if cached:
                cache_hits += 1
            else:
                uncached_decisions += 1
            decisions.append(DecisionRecord(
                scenario=scenario,
                episode=episode_idx,
                step=step_i,
                agent_id=aid,
                food_N=round(percept[0], 4),
                food_S=round(percept[1], 4),
                food_E=round(percept[2], 4),
                food_W=round(percept[3], 4),
                occ_N=int(percept[4] > 0.5),
                occ_S=int(percept[5] > 0.5),
                occ_E=int(percept[6] > 0.5),
                occ_W=int(percept[7] > 0.5),
                food_here=round(percept[8], 4),
                energy=round(percept[9], 4),
                free_cells=free_cells,
                move=move_name,
                reproduce=reproduce_intended,
                did_reproduce=did_reproduce,
                did_move=did_move,
                cached=cached,
                error=str(error) if error else None,
            ))

        if n == 0:
            n_steps_alive = step_i + 1
            pop_history.extend([0] * (MAX_STEPS - len(pop_history)))
            move_freq_history.extend([0.0] * (MAX_STEPS - len(move_freq_history)))
            harvested_food_history.extend([0.0] * (MAX_STEPS - len(harvested_food_history)))
            births_history.extend([0] * (MAX_STEPS - len(births_history)))
            entropy_history.extend([0.0] * (MAX_STEPS - len(entropy_history)))
            confidence_history.extend([1.0] * (MAX_STEPS - len(confidence_history)))
            break

        step_i += 1

    pop_final = int(pop_history[-1]) if pop_history else 0
    pop_max = int(max(pop_history)) if pop_history else 0
    extinct = 1 if pop_final == 0 else 0
    avg_last50 = (
        float(np.mean(pop_history[-50:]))
        if pop_history
        else 0.0
    )

    # Aggregate new metrics
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

    return TestResult(
        pop_final=pop_final,
        avg_pop_last50=avg_last50,
        pop_max=pop_max,
        extinct=extinct,
        total_moves=total_moves,
        pop_history=[float(v) for v in pop_history],
        n_steps_alive=n_steps_alive,
        movement_frequency=movement_frequency,
        avg_harvested_food=avg_harvested_food,
        reproduction_events=total_births,
        action_entropy=action_entropy,
        policy_confidence=policy_confidence,
        move_freq_history=move_freq_history,
        harvested_food_history=harvested_food_history,
        births_history=births_history,
        entropy_history=entropy_history,
        confidence_history=confidence_history,
        decisions=decisions,
        technical_failures=technical_failures,
        cache_hits=cache_hits,
        uncached_decisions=uncached_decisions,
        errors=errors,
    )


def _worker_episode(
    args: Tuple[str, int, str, int],
) -> Tuple[str, int, int, TestResult, List[RejectedEpisodeAttempt]]:
    torch.set_num_threads(1)

    scenario, episode_idx, prompt_variant, first_attempt = args
    params = get_params_for_scenario(scenario)
    params.llm_prompt_variant = prompt_variant
    rejected_attempts: List[RejectedEpisodeAttempt] = []

    for attempt in range(first_attempt, first_attempt + MAX_EPISODE_RESTARTS + 1):
        result = _run_episode(params, scenario, episode_idx, attempt)
        if not (REJECT_FAILED_EPISODES and result.technical_failures > 0):
            return scenario, episode_idx, attempt, result, rejected_attempts

        rejected_attempts.append(
            RejectedEpisodeAttempt(
                attempt=attempt,
                steps_completed=result.n_steps_alive,
                decisions_before_rejection=len(result.decisions),
                technical_failures=result.technical_failures,
                errors=sorted(set(result.errors)),
            )
        )
        if attempt < first_attempt + MAX_EPISODE_RESTARTS:
            print(
                "[AUTO-RESTART] "
                f"{scenario} · episode {episode_idx} · attempt {attempt} "
                "discarded after an LLM/API failure; restarting the whole "
                "episode automatically.",
                flush=True,
            )

    return scenario, episode_idx, attempt, result, rejected_attempts


def _aggregate_results(
    results_by_scenario: Dict[str, List[TestResult]],
    ts_pop_sum_by_scenario: Optional[Dict[str, np.ndarray]] = None,
    ts_pop_count_by_scenario: Optional[Dict[str, np.ndarray]] = None,
) -> Tuple[List[List], List[List]]:
    summary_rows: List[List] = []
    ts_rows: List[List] = []
    ts_pop_sum_by_scenario = ts_pop_sum_by_scenario or {}
    ts_pop_count_by_scenario = ts_pop_count_by_scenario or {}

    for scenario, episode_results in results_by_scenario.items():
        if not episode_results:
            continue

        final_list_global: List[int] = []
        final_list_survived: List[int] = []
        last50_list_global: List[float] = []
        last50_list_survived: List[float] = []
        max_pop_list: List[int] = []
        steps_alive_list: List[int] = []
        ext_list: List[int] = []
        move_list: List[int] = []
        # new metrics lists
        mf_list: List[float] = []
        hf_list: List[float] = []
        repro_list: List[int] = []
        entropy_list: List[float] = []
        confidence_list: List[float] = []

        pop_sum_by_step = np.array(
            ts_pop_sum_by_scenario.get(scenario, np.zeros(MAX_STEPS, dtype=np.float64)),
            dtype=np.float64,
            copy=True,
        )
        pop_count_by_step = np.array(
            ts_pop_count_by_scenario.get(scenario, np.zeros(MAX_STEPS, dtype=np.int32)),
            dtype=np.int32,
            copy=True,
        )
        # TS accumulators for new metrics
        mf_sum_by_step = np.zeros(MAX_STEPS, dtype=np.float64)
        hf_sum_by_step = np.zeros(MAX_STEPS, dtype=np.float64)
        births_sum_by_step = np.zeros(MAX_STEPS, dtype=np.float64)
        ent_sum_by_step = np.zeros(MAX_STEPS, dtype=np.float64)
        conf_sum_by_step = np.zeros(MAX_STEPS, dtype=np.float64)

        for res in episode_results:
            final_list_global.append(res.pop_final)
            last50_list_global.append(res.avg_pop_last50)
            max_pop_list.append(res.pop_max)
            steps_alive_list.append(res.n_steps_alive)
            ext_list.append(res.extinct)
            move_list.append(res.total_moves)
            mf_list.append(res.movement_frequency)
            hf_list.append(res.avg_harvested_food)
            repro_list.append(res.reproduction_events)
            entropy_list.append(res.action_entropy)
            confidence_list.append(res.policy_confidence)

            if res.extinct == 0:
                final_list_survived.append(res.pop_final)
                last50_list_survived.append(res.avg_pop_last50)

            if res.pop_history:
                for t in range(res.n_steps_alive):
                    pop_sum_by_step[t] += float(res.pop_history[t])
                    pop_count_by_step[t] += 1

            # Accumulate new metric timeseries
            for t in range(min(res.n_steps_alive, len(res.move_freq_history))):
                mf_sum_by_step[t] += float(res.move_freq_history[t])
                hf_sum_by_step[t] += float(res.harvested_food_history[t])
                births_sum_by_step[t] += float(res.births_history[t])
                ent_sum_by_step[t] += float(res.entropy_history[t])
                conf_sum_by_step[t] += float(res.confidence_history[t])

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

        summary_rows.append(
            [
                scenario,
                0,
                "LLM",
                float(np.mean(final_list_global)),
                float(np.mean(final_list_survived)) if final_list_survived else float("nan"),
                float(np.mean(last50_list_global)),
                float(np.mean(last50_list_survived)) if last50_list_survived else float("nan"),
                float(np.mean(max_pop_list)),
                float(np.mean(steps_alive_list)),
                float(np.mean(move_list)),
                float(np.std(final_list_global, ddof=1)) if len(final_list_global) > 1 else float("nan"),
                float(np.std(final_list_survived, ddof=1)) if len(final_list_survived) > 1 else float("nan"),
                float(np.std(last50_list_global, ddof=1)) if len(last50_list_global) > 1 else float("nan"),
                float(np.std(last50_list_survived, ddof=1)) if len(last50_list_survived) > 1 else float("nan"),
                float(np.std(move_list, ddof=1)) if len(move_list) > 1 else float("nan"),
                int(np.sum(ext_list)),
                float(np.mean(ext_list)) if ext_list else float("nan"),
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
            ]
        )

        for step in range(MAX_STEPS):
            if pop_count_by_step[step] == 0:
                continue
            nc = pop_count_by_step[step]
            ts_rows.append(
                [
                    scenario,
                    0,
                    "LLM",
                    step,
                    float(pop_sum_by_step[step] / nc),
                    float("nan"),
                    float("nan"),
                    int(nc),
                    0,
                    float(mf_sum_by_step[step] / nc) if nc > 0 else 0.0,
                    float(hf_sum_by_step[step] / nc) if nc > 0 else 0.0,
                    float(births_sum_by_step[step] / nc) if nc > 0 else 0.0,
                    float(ent_sum_by_step[step] / nc) if nc > 0 else 0.0,
                    float(conf_sum_by_step[step] / nc) if nc > 0 else 0.0,
                ]
            )

    summary_rows.sort(key=lambda row: (row[0], row[1], row[2]))
    ts_rows.sort(key=lambda row: (row[0], row[1], row[2], row[3]))
    return summary_rows, ts_rows


def _build_episode_tasks(
    next_episode_by_scenario: Dict[str, int],
    episodes_to_add_by_scenario: Dict[str, int],
) -> List[Tuple[str, int, str, int]]:
    tasks: List[Tuple[str, int, str, int]] = []
    for scenario in VALID_SCENARIOS:
        start_idx = next_episode_by_scenario.get(scenario, 0)
        for offset in range(episodes_to_add_by_scenario.get(scenario, 0)):
            episode_idx = start_idx + offset
            tasks.append((scenario, episode_idx, PROMPT_VARIANT, 0))
    return tasks


def _append_simulation_row(scenario: str, episode_idx: int, res: TestResult) -> None:
    with open(SIM_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                scenario,
                episode_idx,
                0,
                "LLM",
                res.pop_final,
                res.avg_pop_last50,
                res.pop_max,
                res.extinct,
                res.total_moves,
                res.n_steps_alive,
                json.dumps(res.pop_history, separators=(",", ":")),
                round(res.movement_frequency, 6),
                round(res.avg_harvested_food, 6),
                res.reproduction_events,
                round(res.action_entropy, 6),
                round(res.policy_confidence, 6),
                res.technical_failures,
                res.cache_hits,
                res.uncached_decisions,
            ]
        )
        f.flush()
        os.fsync(f.fileno())


def _append_decisions(decisions: List[DecisionRecord]) -> None:
    """Append decision records to the decisions CSV."""
    if not decisions:
        return
    with open(DECISIONS_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        for d in decisions:
            writer.writerow([
                d.scenario,
                d.episode,
                d.step,
                d.agent_id,
                d.food_N,
                d.food_S,
                d.food_E,
                d.food_W,
                d.occ_N,
                d.occ_S,
                d.occ_E,
                d.occ_W,
                d.food_here,
                d.energy,
                d.free_cells,
                d.move,
                d.reproduce,
                d.did_reproduce,
                d.did_move,
                d.cached,
                d.error or "",
            ])
        f.flush()
        os.fsync(f.fileno())


def _write_simulation_header() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(SIM_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(SIM_HEADER)
        f.flush()
        os.fsync(f.fileno())


def _write_decisions_header() -> None:
    """Write header for the decisions CSV."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(DECISIONS_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(DECISIONS_HEADER)
        f.flush()
        os.fsync(f.fileno())


def _write_rejected_header() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REJECTED_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(REJECTED_HEADER)
        f.flush()
        os.fsync(f.fileno())


def _append_rejected_episode(
    scenario: str, episode_idx: int, rejected: RejectedEpisodeAttempt
) -> None:
    if not REJECTED_CSV.exists() or REJECTED_CSV.stat().st_size == 0:
        _write_rejected_header()
    with open(REJECTED_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            scenario,
            episode_idx,
            rejected.attempt,
            rejected.steps_completed,
            rejected.decisions_before_rejection,
            rejected.technical_failures,
            json.dumps(rejected.errors, separators=(",", ":")),
        ])
        f.flush()
        os.fsync(f.fileno())


def _ensure_simulation_header_for_append() -> None:
    if not SIM_CSV.exists() or SIM_CSV.stat().st_size == 0:
        _write_simulation_header()
        return

    with open(SIM_CSV, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames == SIM_HEADER:
            return
        rows = list(reader)

    tmp_path = SIM_CSV.with_suffix(".tmp")
    with open(tmp_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SIM_HEADER)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in SIM_HEADER})
        f.flush()
        os.fsync(f.fileno())
    tmp_path.replace(SIM_CSV)


def _ensure_decisions_header_for_append() -> None:
    """Add missing optional columns without discarding existing decisions."""
    if not DECISIONS_CSV.exists() or DECISIONS_CSV.stat().st_size == 0:
        _write_decisions_header()
        return
    with open(DECISIONS_CSV, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        if fieldnames == DECISIONS_HEADER:
            return
        required_columns = DECISIONS_HEADER[:-2]
        if fieldnames != required_columns:
            raise ValueError(
                "The existing decision CSV has incompatible columns and cannot "
                f"be extended safely. Found columns: {fieldnames!r}"
            )

    tmp_path = DECISIONS_CSV.with_suffix(DECISIONS_CSV.suffix + ".tmp")
    try:
        with open(DECISIONS_CSV, newline="") as source, open(
            tmp_path, "w", newline=""
        ) as destination:
            reader = csv.DictReader(source)
            writer = csv.DictWriter(destination, fieldnames=DECISIONS_HEADER)
            writer.writeheader()
            for row in reader:
                migrated = {field: row.get(field, "") for field in DECISIONS_HEADER}
                writer.writerow(migrated)
            destination.flush()
            os.fsync(destination.fileno())
        tmp_path.replace(DECISIONS_CSV)
    except BaseException:
        if tmp_path.exists():
            tmp_path.unlink()
        raise

    console.print(
        "[yellow]Decision CSV updated without deleting existing rows.[/yellow]"
    )


def _initialize_outputs() -> None:
    _write_simulation_header()
    _write_decisions_header()
    _write_rejected_header()
    _write_outputs([], [])


def _empty_run_state() -> ExistingRunState:
    return ExistingRunState(
        results_by_scenario={scenario: [] for scenario in VALID_SCENARIOS},
        next_episode_by_scenario={scenario: 0 for scenario in VALID_SCENARIOS},
        ts_pop_sum_by_scenario={},
        ts_pop_count_by_scenario={},
    )


def _load_existing_run_state() -> ExistingRunState:
    state = _empty_run_state()
    max_episode_by_scenario = {scenario: -1 for scenario in VALID_SCENARIOS}

    if SIM_CSV.exists() and SIM_CSV.stat().st_size > 0:
        with open(SIM_CSV, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                scenario = (row.get("scenario") or "").strip()
                if scenario not in state.results_by_scenario:
                    continue
                try:
                    episode_idx = int(row.get("episode") or 0)
                    state.results_by_scenario[scenario].append(
                        TestResult(
                            pop_final=int(float(row["pop_final"])),
                            avg_pop_last50=float(row["avg_pop_last50"]),
                            pop_max=int(float(row["pop_max"])),
                            extinct=int(float(row["extinct"])),
                            total_moves=int(float(row["total_moves"])),
                            pop_history=[],
                            n_steps_alive=int(float(row["n_steps_alive"])),
                            movement_frequency=float(row.get("movement_frequency", 0)),
                            avg_harvested_food=float(row.get("avg_harvested_food", 0)),
                            reproduction_events=int(float(row.get("reproduction_events", 0))),
                            action_entropy=float(row.get("action_entropy", 0)),
                            policy_confidence=float(row.get("policy_confidence", 0)),
                        )
                    )
                    max_episode_by_scenario[scenario] = max(max_episode_by_scenario[scenario], episode_idx)
                except (KeyError, TypeError, ValueError):
                    continue

    for scenario in VALID_SCENARIOS:
        state.next_episode_by_scenario[scenario] = max_episode_by_scenario[scenario] + 1

    if TS_CSV.exists() and TS_CSV.stat().st_size > 0:
        with open(TS_CSV, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                scenario = (row.get("scenario") or "").strip()
                if scenario not in state.results_by_scenario:
                    continue
                try:
                    step = int(float(row["step"]))
                    if step < 0 or step >= MAX_STEPS:
                        continue
                    n_samples = int(float(row["n_pop_samples"]))
                    avg_pop = float(row["avg_pop_step"])
                except (KeyError, TypeError, ValueError):
                    continue
                if scenario not in state.ts_pop_sum_by_scenario:
                    state.ts_pop_sum_by_scenario[scenario] = np.zeros(MAX_STEPS, dtype=np.float64)
                    state.ts_pop_count_by_scenario[scenario] = np.zeros(MAX_STEPS, dtype=np.int32)
                state.ts_pop_sum_by_scenario[scenario][step] = avg_pop * n_samples
                state.ts_pop_count_by_scenario[scenario][step] = n_samples

    return state


def _show_existing_counts(state: ExistingRunState) -> None:
    table = Table(show_header=True, header_style="bold cyan", expand=False)
    table.add_column("Scenario")
    table.add_column("Saved simulations", justify="right")
    table.add_column("Next episode", justify="right")
    for scenario in VALID_SCENARIOS:
        table.add_row(
            scenario,
            str(len(state.results_by_scenario.get(scenario, []))),
            str(state.next_episode_by_scenario.get(scenario, 0)),
        )
    console.print(Panel(table, title="[bold]Existing simulation CSV[/bold]", border_style="dim", expand=False))


def _ask_append_or_rewrite(state: ExistingRunState) -> bool:
    total_existing = sum(len(rows) for rows in state.results_by_scenario.values())
    if total_existing == 0:
        console.print("[dim]No previous LLM simulations found; starting a new CSV.[/dim]")
        return False

    forced_mode = os.getenv("LLM_EXISTING_OUTPUT_MODE", "").strip().lower()
    if forced_mode in {"a", "add", "append"}:
        console.print(
            "[green]Automatically resuming from previously saved valid results.[/green]"
        )
        return True
    if forced_mode in {"r", "rewrite", "zero"}:
        console.print("[yellow]Rewrite mode preselected.[/yellow]")
        return False

    while True:
        try:
            answer = input("  Append simulations to the existing CSV or rewrite from scratch? [a/r] ").strip().lower()
        except EOFError:
            console.print("[yellow]Interactive input unavailable; rewriting from scratch.[/yellow]")
            return False
        if answer in {"a", "add", "append"}:
            return True
        if answer in {"r", "rewrite", "zero"}:
            return False
        console.print("[yellow]Enter 'a' to append or 'r' to rewrite.[/yellow]")


def _ask_episodes_to_add(defaults: Dict[str, int]) -> Dict[str, int]:
    episodes_to_add: Dict[str, int] = {}
    console.print()
    console.print("[bold]How many new simulations should run for each scenario?[/bold]")
    for scenario in VALID_SCENARIOS:
        default = int(defaults.get(scenario, 0))
        while True:
            try:
                raw = input(f"  {scenario} [default {default}]: ").strip()
            except EOFError:
                console.print("[yellow]Interactive input unavailable; using the configured defaults.[/yellow]")
                return {s: int(defaults.get(s, 0)) for s in VALID_SCENARIOS}
            if raw == "":
                value = default
            else:
                try:
                    value = int(raw)
                except ValueError:
                    console.print("[yellow]Enter an integer >= 0.[/yellow]")
                    continue
            if value < 0:
                console.print("[yellow]Enter an integer >= 0.[/yellow]")
                continue
            episodes_to_add[scenario] = value
            break
    return episodes_to_add


def _write_outputs(summary_rows: List[List], ts_rows: List[List]) -> None:
    summary_rows.sort(key=lambda row: (row[0], row[1], row[2]))
    ts_rows.sort(key=lambda row: (row[0], row[1], row[2], row[3]))

    with open(FULL_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(SUMMARY_HEADER)
        writer.writerows(summary_rows)
        f.flush()
        os.fsync(f.fileno())

    with open(TS_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(TS_HEADER)
        writer.writerows(ts_rows)
        f.flush()
        os.fsync(f.fileno())


def main() -> None:
    t_start = time.perf_counter()
    _configure_prompt_variant(_ask_prompt_variant())

    console.print()
    console.print(
        Panel.fit(
            "[bold white]LLM ANALYSIS[/bold white]\n"
            "[dim]Dedicated benchmark with run_analysis-compatible CSV schema[/dim]",
            border_style="bright_green",
            padding=(1, 4),
        )
    )

    cfg_table = Table(show_header=False, box=None, padding=(0, 2))
    cfg_table.add_column(style="bold cyan")
    cfg_table.add_column(style="white")
    cfg_table.add_row(
        "Episodes per scenario",
        ", ".join(f"{s}={SCENARIO_EPISODES.get(s, 1)}" for s in VALID_SCENARIOS),
    )
    cfg_table.add_row("Max steps per episode", str(MAX_STEPS))
    cfg_table.add_row("Episode workers", str(MAX_WORKERS))
    cfg_table.add_row("Abort on terminal API failure", str(REJECT_FAILED_EPISODES))
    cfg_table.add_row("Automatic episode restarts", str(MAX_EPISODE_RESTARTS))
    cfg_table.add_row("Resumes from last valid step", str(MAX_STEP_RESUMES))
    cfg_table.add_row("HTTP timeout", f"{os.getenv('LLM_TIMEOUT_SECONDS', '45')} s")
    cfg_table.add_row("LLM enabled", os.getenv("LLM_ENABLED", "0"))
    cfg_table.add_row("LLM model", LLM_MODEL)
    cfg_table.add_row("Prompt variant", PROMPT_VARIANT)
    cfg_table.add_row("Output", str(FULL_CSV))
    cfg_table.add_row("TS output", str(TS_CSV))
    cfg_table.add_row("Simulation output", str(SIM_CSV))
    cfg_table.add_row("Decisions output", str(DECISIONS_CSV))
    cfg_table.add_row("Rejected output", str(REJECTED_CSV))
    console.print(Panel(cfg_table, title="[bold]Configuration[/bold]", border_style="dim", expand=False))

    existing_state = _load_existing_run_state()
    _show_existing_counts(existing_state)
    append_mode = _ask_append_or_rewrite(existing_state)
    if append_mode:
        _ensure_simulation_header_for_append()
        _ensure_decisions_header_for_append()
        results_by_scenario = existing_state.results_by_scenario
        ts_pop_sum_by_scenario = existing_state.ts_pop_sum_by_scenario
        ts_pop_count_by_scenario = existing_state.ts_pop_count_by_scenario
        next_episode_by_scenario = existing_state.next_episode_by_scenario
        console.print("[green]Appending new simulations to the existing CSV.[/green]")
    else:
        _initialize_outputs()
        empty_state = _empty_run_state()
        results_by_scenario = empty_state.results_by_scenario
        ts_pop_sum_by_scenario = empty_state.ts_pop_sum_by_scenario
        ts_pop_count_by_scenario = empty_state.ts_pop_count_by_scenario
        next_episode_by_scenario = empty_state.next_episode_by_scenario
        console.print("[yellow]Rewriting the output CSV files from scratch.[/yellow]")

    target_raw = os.getenv("LLM_TARGET_EPISODES_PER_SCENARIO", "").strip()
    if target_raw:
        target_per_scenario = max(0, int(target_raw))
        episode_defaults = {
            scenario: max(
                0,
                target_per_scenario
                - len(results_by_scenario.get(scenario, [])),
            )
            for scenario in VALID_SCENARIOS
        }
        console.print(
            f"[cyan]Configured target: {target_per_scenario} valid replicates "
            "per scenario.[/cyan]"
        )
    else:
        episode_defaults = SCENARIO_EPISODES
    episodes_to_add_by_scenario = _ask_episodes_to_add(episode_defaults)
    tasks = _build_episode_tasks(next_episode_by_scenario, episodes_to_add_by_scenario)
    total = len(tasks)
    total_scenarios = len(VALID_SCENARIOS)
    planned = ", ".join(f"{s}={episodes_to_add_by_scenario.get(s, 0)}" for s in VALID_SCENARIOS)
    console.print()
    console.print(f"  [bold]New simulations by scenario:[/bold] [yellow]{planned}[/yellow]")
    console.print(
        f"  [bold]Total scenarios:[/bold] [yellow]{total_scenarios}[/yellow]  "
        f"[dim]({total:,} new episodes in this run)[/dim]"
    )
    console.print()

    ctx = mp.get_context("spawn")
    summary_rows, ts_rows = _aggregate_results(
        results_by_scenario,
        ts_pop_sum_by_scenario,
        ts_pop_count_by_scenario,
    )
    if append_mode:
        _write_outputs(summary_rows, ts_rows)

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
        task_id = progress.add_task("LLM benchmark in progress...", total=total)

        with ctx.Pool(processes=MAX_WORKERS) as pool:
            for scenario, episode_idx, attempt, res, rejected_attempts in pool.imap_unordered(
                _worker_episode, tasks
            ):
                for rejected in rejected_attempts:
                    _append_rejected_episode(scenario, episode_idx, rejected)

                if REJECT_FAILED_EPISODES and res.technical_failures > 0:
                    error_label = "; ".join(sorted(set(res.errors)))
                    progress.console.print(
                        "  [bold red]✗ Automatic restarts exhausted[/bold red] "
                        f"{scenario} · episode {episode_idx} · "
                        f"{res.technical_failures} failure(s): {error_label}"
                    )
                    raise RuntimeError(
                        "Terminal LLM/API failure persisted after all automatic "
                        "episode restarts. Every invalid attempt was excluded; "
                        "only previously completed healthy episodes remain in "
                        "the canonical CSV files."
                    )

                results_by_scenario[scenario].append(res)
                _append_simulation_row(scenario, episode_idx, res)
                _append_decisions(res.decisions)
                summary_rows, ts_rows = _aggregate_results(
                    results_by_scenario,
                    ts_pop_sum_by_scenario,
                    ts_pop_count_by_scenario,
                )
                _write_outputs(summary_rows, ts_rows)
                progress.console.print(
                    "  [bold green]✓ Valid replicate saved[/bold green] "
                    f"{scenario} · episode {episode_idx} · "
                    f"{len(res.decisions):,} exact decisions"
                    + (
                        f" · completed after {len(rejected_attempts)} automatic restart(s)"
                        if rejected_attempts
                        else ""
                    )
                )
                progress.advance(task_id)

        progress.update(task_id, description="[bold bright_green]LLM benchmark completed!")

    elapsed = time.perf_counter() - t_start
    minutes, seconds = divmod(int(elapsed), 60)

    console.print()
    summary = Table.grid(padding=(0, 2))
    summary.add_column(style="bold")
    summary.add_column()
    summary.add_row("[green]Rows written[/green]", f"{len(summary_rows)}")
    summary.add_row("[green]Timeseries rows[/green]", f"{len(ts_rows)}")
    summary.add_row("[green]Simulation rows[/green]", f"{sum(len(v) for v in results_by_scenario.values())}")
    total_decisions = sum(len(res.decisions) for vals in results_by_scenario.values() for res in vals)
    summary.add_row("[green]Decision rows[/green]", f"{total_decisions:,}")
    summary.add_row("[cyan]Total time[/cyan]", f"{minutes}m {seconds}s")
    summary.add_row("[blue]Output[/blue]", str(FULL_CSV))
    summary.add_row("[blue]TS output[/blue]", str(TS_CSV))
    summary.add_row("[blue]Simulation output[/blue]", str(SIM_CSV))
    summary.add_row("[blue]Decisions output[/blue]", str(DECISIONS_CSV))
    summary.add_row("[blue]Rejected output[/blue]", str(REJECTED_CSV))

    console.print(
        Panel(
            summary,
            title="[bold bright_green]LLM benchmark completed[/bold bright_green]",
            border_style="bright_green",
            expand=False,
        )
    )
    console.print()


if __name__ == "__main__":
    setup_logging()
    mp.freeze_support()
    main()
