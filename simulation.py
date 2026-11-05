"""Headless simulation, multiprocessing batch runner, and weight-loading utilities."""

import csv
import itertools
import multiprocessing
import random
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from config import (
    MLP_OFFLINE_INIT,
    MLP_ONLINE_LEARNING,
    RULE_VARIANT_TAGS,
    VALID_SCENARIOS,
    WEIGHTS_DIR,
    RESULTS_DIR,
    ModelParams,
    get_params_for_scenario,
)
from model import World
from network import NeuralNetwork

# alpha-gene sweep values (0.0 -> 2.0, step 0.1)
GENE_SWEEP_VALUES: List[float] = [round(i * 0.1, 1) for i in range(21)]

GRID_SEARCH_PARAMS: Dict[str, List[float]] = {
    "food_regen":    [0.1, 0.3, 0.5],
    "initial_alpha": [0.5, 1.0, 1.5],
}

_WORKER_NET_CACHE: Dict[Tuple[str, int, str], Optional[NeuralNetwork]] = {}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _normalize_weight_tag(tag: str) -> str:
    return tag.strip().lower()


def _resolve_rb_variant(
    variant: str,
) -> Tuple[Optional[float], Optional[float], bool, bool, bool]:
    """Return (fixed_alpha, fixed_beta, mutate_propensity, random_offspring, propensity_learning)."""
    if variant == "rb_rand_fix":
        # Each founder samples alpha and beta once. Offspring inherit both
        # values exactly, so genes remain fixed along each lineage.
        return None, None, False, False, False
    if variant == "rb_rand_rand":
        return None, None, False, True, False
    if variant == "rb_rand_evo":
        return None, None, True, False, False
    if variant == "rb_rand_learn":
        return None, None, False, False, True
    if variant == "rb_fix_fix":
        return 1.0, 0.5, False, False, False
    if variant == "rb_fix_evo":
        return 1.0, 0.5, True, False, False
    if variant == "rb_fix_learn":
        return 1.0, 0.5, False, False, True
    raise ValueError(f"Unknown rule-based variant: {variant}")


def load_weights(
    scenario: str,
    n_layers: int,
    weight_tag: str = "sup_rb_fix_fix_auto",
) -> Optional[NeuralNetwork]:
    """Load pre-trained weights for the scenario from the weights directory."""
    weight_tag = _normalize_weight_tag(weight_tag)
    path = WEIGHTS_DIR / f"weights_{scenario}_L{n_layers}_{weight_tag}.pth"
    if not path.exists():
        print(f"[WARN] Weights not found: {path}. Using random network.")
        return None
    net = NeuralNetwork(num_hidden_layers=n_layers)
    try:
        net.load_state_dict(torch.load(path, weights_only=True))
        net.eval()
        print(f"[INFO] Weights loaded: {path.name}")
        return net
    except Exception as exc:
        print(f"[ERR] Failed to load {path}: {exc}")
        return None


def _get_worker_net_template(
    scenario: str,
    n_layers: int,
    weight_tag: str,
) -> Optional[NeuralNetwork]:
    """Load and cache weights once per worker process."""
    tag = _normalize_weight_tag(weight_tag)
    key = (scenario, n_layers, tag)
    if key in _WORKER_NET_CACHE:
        return _WORKER_NET_CACHE[key]

    fname = WEIGHTS_DIR / f"weights_{scenario}_L{n_layers}_{tag}.pth"
    if not fname.exists():
        _WORKER_NET_CACHE[key] = None
        return None

    net = NeuralNetwork(num_hidden_layers=n_layers)
    net.load_state_dict(torch.load(fname, weights_only=True))
    net.eval()
    _WORKER_NET_CACHE[key] = net
    return net


# ---------------------------------------------------------------------------
# Headless simulation
# ---------------------------------------------------------------------------

@dataclass
class SimResult:
    mean_pop: float
    mean_pop_last50: float
    max_pop: int
    extinction_step: int
    final_pop: int
    extinct: int
    total_moves: int
    mean_food: float


def run_headless(
    params: ModelParams,
    mode: str,
    steps: int,
    seed: int,
    net_template: Optional[NeuralNetwork] = None,
    mutate_propensity: bool = True,
    random_offspring_propensity: bool = False,
    propensity_learning: bool = False,
    net_learning: bool = False,
) -> SimResult:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    world = World(
        params,
        agent_type=mode,
        net_template=net_template,
        mutate_propensity=mutate_propensity,
        random_offspring_propensity=random_offspring_propensity,
        propensity_learning=propensity_learning,
        net_learning=net_learning,
    )

    n_steps = 0
    pop_sum = 0.0
    food_sum = 0.0
    max_pop = 0
    final_pop = 0
    pop_last50: deque[int] = deque(maxlen=50)

    total_moves = 0
    extinction_step = steps

    for t in range(steps):
        world.step()
        n = len(world.agents)
        n_steps += 1
        pop_sum += n
        food_sum += world.total_food
        if n > max_pop:
            max_pop = n
        final_pop = n
        pop_last50.append(n)
        total_moves += world.moves_step
        if n == 0:
            extinction_step = t
            break

    world.flush_online_learning()
    if n_steps == 0:
        return SimResult(0, 0.0, 0, 0, 0, 1, 0, 0.0)

    mean_pop = pop_sum / n_steps
    mean_pop_last50 = sum(pop_last50) / len(pop_last50)
    return SimResult(
        mean_pop=mean_pop,
        mean_pop_last50=mean_pop_last50,
        max_pop=max_pop,
        extinction_step=extinction_step,
        final_pop=final_pop,
        extinct=int(final_pop == 0),
        total_moves=total_moves,
        mean_food=food_sum / n_steps,
    )


# ---------------------------------------------------------------------------
# Multiprocessing workers
# ---------------------------------------------------------------------------

def _worker_task(args: tuple) -> List:
    (
        scenario_base, param_override, mode, steps, replica, seed,
        n_layers, weight_tag, rule_variant,
        prop_fixed_rb, beta_fixed_rb, mutate_prop_rb, random_offspring_rb, propensity_learning_rb,
    ) = args

    params = get_params_for_scenario(scenario_base)
    params.net_hidden_layers = n_layers
    for k, v in param_override.items():
        if hasattr(params, k):
            setattr(params, k, v)
    if mode == "rule" and prop_fixed_rb is not None:
        params.initial_alpha = float(prop_fixed_rb)
    if mode == "rule" and beta_fixed_rb is not None:
        params.initial_beta = float(beta_fixed_rb)

    net_tpl = None
    if mode in MLP_OFFLINE_INIT:
        net_tpl = _get_worker_net_template(scenario_base, n_layers, weight_tag)

    res = run_headless(
        params, mode, steps, seed, net_tpl,
        mutate_propensity=(mutate_prop_rb if mode == "rule" else True),
        random_offspring_propensity=(random_offspring_rb if mode == "rule" else False),
        propensity_learning=(propensity_learning_rb if mode == "rule" else False),
        net_learning=(mode in MLP_ONLINE_LEARNING),
    )
    return list(param_override.values()) + [
        replica, res.mean_pop, res.max_pop, res.extinction_step,
        res.final_pop, res.total_moves, res.mean_food,
    ]


def _gene_sweep_worker(args: tuple) -> Dict[str, object]:
    scenario, alpha, replica, steps, seed = args
    params = get_params_for_scenario(scenario)
    params.initial_alpha = float(alpha)
    res = run_headless(params, "rule", steps, seed,
                       mutate_propensity=False, random_offspring_propensity=False)
    return {
        "scenario": scenario,
        "base_metabolism": params.base_metabolism,
        "food_regen": params.food_regen,
        "alpha_gene": float(alpha),
        "replica": int(replica),
        "seed": int(seed),
        "steps": int(steps),
        "final_pop": int(res.final_pop),
        "mean_pop_last50": float(res.mean_pop_last50),
        "extinct": int(res.extinct),
        "extinction_step": int(res.extinction_step),
    }


# ---------------------------------------------------------------------------
# CSV utility
# ---------------------------------------------------------------------------

def _write_csv_dicts(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _progress_bar(done: int, total: int) -> None:
    if total <= 0:
        return
    perc = done / total * 100
    filled = int(perc / 100 * 30)
    bar = "█" * filled + "░" * (30 - filled)
    sys.stdout.write(
        f"\r  \033[36m{bar}\033[0m \033[1m{perc:5.1f}%\033[0m ({done}/{total})"
    )
    sys.stdout.flush()


def _print_header(title: str) -> None:
    w = 54
    print("\n" + "\033[36m" + "╔" + "═" * w + "╗")
    print("║" + title.center(w) + "║")
    print("╚" + "═" * w + "╝" + "\033[0m")


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def run_batch(
    scenario: str,
    mode: str,
    steps: int,
    replicas: int,
    n_layers: int,
    weight_tag: str = "sup_rb_fix_fix_auto",
    rule_variant: str = "rb_fix_fix",
) -> None:
    _print_header(f"BATCH: {scenario.upper()} [{mode}]")

    prop_fixed_rb: Optional[float] = None
    beta_fixed_rb: Optional[float] = None
    mutate_prop_rb, random_offspring_rb, propensity_learning_rb = True, False, False
    if mode == "rule":
        prop_fixed_rb, beta_fixed_rb, mutate_prop_rb, random_offspring_rb, propensity_learning_rb = (
            _resolve_rb_variant(rule_variant)
        )
        print(f"[INFO] Rule variant: {rule_variant}")
    elif mode == "llm":
        print("[INFO] LLM agent: set LLM_API_KEY/OPENAI_API_KEY and LLM_MODEL to enable API decisions.")
        print("[INFO] Set LLM_ENABLED=0 to force the local fallback policy.")

    grid_params = {k: v for k, v in GRID_SEARCH_PARAMS.items()
                   if not (mode == "rule" and k == "initial_alpha")}
    keys = list(grid_params.keys())
    combinations = list(itertools.product(*grid_params.values()))

    base_seed = int(time.time())
    total_sims = len(combinations) * replicas

    def _iter_tasks():
        for combo_idx, combo in enumerate(combinations):
            override = dict(zip(keys, combo))
            for r in range(replicas):
                seed = base_seed + combo_idx * 100_000 + r
                yield (
                    scenario,
                    override,
                    mode,
                    steps,
                    r,
                    seed,
                    n_layers,
                    weight_tag,
                    rule_variant,
                    prop_fixed_rb,
                    beta_fixed_rb,
                    mutate_prop_rb,
                    random_offspring_rb,
                    propensity_learning_rb,
                )

    print(f"\n  \033[37mSimulations: \033[1;33m{total_sims}\033[0m"
          f"  \033[37mCores: \033[1;33m{multiprocessing.cpu_count()}\033[0m\n")

    if mode == "rule":
        fname = f"batch_{scenario}_{mode}_{rule_variant}.csv"
    elif mode in MLP_OFFLINE_INIT:
        fname = f"batch_{scenario}_{mode}_{weight_tag}.csv"
    else:
        fname = f"batch_{scenario}_{mode}.csv"

    out_path = RESULTS_DIR / fname
    out_path.parent.mkdir(parents=True, exist_ok=True)
    header = keys + ["replica", "mean_pop", "max_pop", "extinction_step",
                     "final_pop", "total_moves", "mean_food"]

    start = time.time()
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        if total_sims > 0:
            workers = multiprocessing.cpu_count()
            chunksize = max(1, total_sims // max(1, workers * 8))
            with multiprocessing.Pool() as pool:
                for i, row in enumerate(pool.imap_unordered(_worker_task, _iter_tasks(), chunksize=chunksize)):
                    writer.writerow(row)
                    if (i + 1) % 10 == 0 or (i + 1) == total_sims:
                        _progress_bar(i + 1, total_sims)

    mins, secs = divmod(int(time.time() - start), 60)
    print(f"\n\n  \033[1;32mDone!\033[0m  {out_path}")
    print(f"  \033[37mTime: {mins}m {secs}s\033[0m")


def run_rb_gene_sweep_batch(
    steps: int,
    replicas: int,
    scenarios: Optional[List[str]] = None,
) -> None:
    scenario_list = scenarios or list(VALID_SCENARIOS)
    _print_header("BATCH: RB FIXED ALPHA GENE SWEEP")
    print(f"[INFO] Scenarios: {', '.join(scenario_list)}")
    print(f"[INFO] Gene: {GENE_SWEEP_VALUES[0]:.1f} -> {GENE_SWEEP_VALUES[-1]:.1f} (step 0.1)")
    print(f"[INFO] Replicas: {replicas} | Steps: {steps}")

    base_seed = int(time.time())
    total = len(scenario_list) * len(GENE_SWEEP_VALUES) * replicas

    def _iter_gene_tasks():
        for si, s in enumerate(scenario_list):
            for pi, p in enumerate(GENE_SWEEP_VALUES):
                for r in range(replicas):
                    yield (s, p, r, steps, base_seed + si * 100_000 + pi * 1_000 + r)

    print(f"\n  \033[37mSimulations: \033[1;33m{total}\033[0m"
          f"  \033[37mCores: \033[1;33m{multiprocessing.cpu_count()}\033[0m\n")

    detail_rows: List[Dict[str, object]] = []
    start = time.time()
    workers = multiprocessing.cpu_count()
    chunksize = max(1, total // max(1, workers * 8))
    with multiprocessing.Pool() as pool:
        for i, row in enumerate(pool.imap_unordered(_gene_sweep_worker, _iter_gene_tasks(), chunksize=chunksize)):
            detail_rows.append(row)
            if (i + 1) % 10 == 0 or (i + 1) == total:
                _progress_bar(i + 1, total)

    detail_rows.sort(key=lambda r: (str(r["scenario"]), float(r["alpha_gene"]), int(r["replica"])))

    grouped_rows: Dict[Tuple[str, float], List[Dict[str, object]]] = {}
    for row in detail_rows:
        key = (str(row["scenario"]), round(float(row["alpha_gene"]), 6))
        grouped_rows.setdefault(key, []).append(row)

    summary_rows: List[Dict[str, object]] = []
    for scenario in scenario_list:
        params = get_params_for_scenario(scenario)
        for alpha in GENE_SWEEP_VALUES:
            rows = grouped_rows.get((scenario, round(float(alpha), 6)), [])
            if not rows:
                continue
            final_pops = [int(r["final_pop"]) for r in rows]
            pop_last50 = [float(r["mean_pop_last50"]) for r in rows]
            extincts = [int(r["extinct"]) for r in rows]
            summary_rows.append({
                "scenario": scenario,
                "base_metabolism": params.base_metabolism,
                "food_regen": params.food_regen,
                "alpha_gene": alpha,
                "replicas": len(rows),
                "steps": steps,
                "final_pop_mean": float(sum(final_pops) / len(final_pops)),
                "final_pop_std": float(np.std(final_pops)),
                "mean_pop_last50_mean": float(sum(pop_last50) / len(pop_last50)),
                "mean_pop_last50_std": float(np.std(pop_last50)),
                "extinction_rate": float(sum(extincts) / len(extincts)),
            })

    _write_csv_dicts(RESULTS_DIR / "batch_rb_gene_sweep_detail.csv", detail_rows)
    _write_csv_dicts(RESULTS_DIR / "batch_rb_gene_sweep_summary.csv", summary_rows)

    mins, secs = divmod(int(time.time() - start), 60)
    print(f"\n\n  \033[1;32mDone!\033[0m  Time: {mins}m {secs}s")
