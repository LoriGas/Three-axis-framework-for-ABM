"""Empirical sensitivity analysis for online-learning hyperparameters.

The score uses ecological outcomes only (survival and late population), never
the reward being tuned. Common random seeds reduce between-configuration noise.
"""

from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from config import RESULTS_DIR, get_params_for_scenario
from simulation import _get_worker_net_template, run_headless


SCENARIOS = ["S2", "S3", "S4"]
MODES = ["rule_learn", "mlp_learn", "mlp_offline_learn"]


@dataclass(frozen=True)
class Candidate:
    name: str
    birth_bonus: float = 0.20
    survival_bonus: float = 0.02
    energy_scale: float = 3.0
    n_steps: int = 4
    gamma: float = 0.95


def candidates() -> List[Candidate]:
    """One-factor-at-a-time design around the current baseline."""
    return [
        Candidate("baseline"),
        Candidate("birth_0.10", birth_bonus=0.10),
        Candidate("birth_0.40", birth_bonus=0.40),
        Candidate("survival_0.00", survival_bonus=0.00),
        Candidate("survival_0.05", survival_bonus=0.05),
        Candidate("energy_scale_1.5", energy_scale=1.5),
        Candidate("energy_scale_6.0", energy_scale=6.0),
        Candidate("n_steps_2", n_steps=2),
        Candidate("n_steps_8", n_steps=8),
        Candidate("gamma_0.80", gamma=0.80),
        Candidate("gamma_0.99", gamma=0.99),
    ]


def _worker(task: Tuple[Candidate, str, str, int, int]) -> Dict[str, object]:
    candidate, scenario, mode, replica, steps = task
    params = get_params_for_scenario(scenario)
    params.rl_birth_bonus = candidate.birth_bonus
    params.rl_survival_bonus = candidate.survival_bonus
    params.rl_energy_scale = candidate.energy_scale
    params.rl_n_steps = candidate.n_steps
    params.rl_gamma = candidate.gamma

    # Common seed for every candidate at a given scenario/mode/replica.
    mode_offset = {
        "rule_learn": 0,
        "mlp_learn": 100_000,
        "mlp_offline_learn": 200_000,
    }[mode]
    seed = 20_000 + SCENARIOS.index(scenario) * 10_000 + mode_offset + replica

    if mode == "rule_learn":
        result = run_headless(
            params,
            "rule",
            steps,
            seed,
            mutate_propensity=False,
            random_offspring_propensity=False,
            propensity_learning=True,
            net_learning=False,
        )
    elif mode == "mlp_learn":
        result = run_headless(
            params,
            "mlp_rand_learn",
            steps,
            seed,
            mutate_propensity=False,
            random_offspring_propensity=False,
            propensity_learning=False,
            net_learning=True,
        )
    else:
        net_template = _get_worker_net_template(
            scenario, 1, "sup_rb_evo_top5_auto"
        )
        if net_template is None:
            raise FileNotFoundError(f"Offline weights missing for {scenario} L1")
        result = run_headless(
            params,
            "mlp_offline_learn",
            steps,
            seed,
            net_template=net_template,
            mutate_propensity=False,
            random_offspring_propensity=False,
            propensity_learning=False,
            net_learning=True,
        )

    return {
        **asdict(candidate),
        "scenario": scenario,
        "mode": mode,
        "replica": replica,
        "seed": seed,
        "steps": steps,
        "mean_pop_last50": result.mean_pop_last50,
        "final_pop": result.final_pop,
        "max_pop": result.max_pop,
        "extinct": result.extinct,
        "extinction_step": result.extinction_step,
        "total_moves": result.total_moves,
    }


def _write_rows(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _summarise(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    # Aggregate each candidate within scenario and agent family.
    grouped: Dict[Tuple[str, str, str], List[Dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault((str(row["name"]), str(row["scenario"]), str(row["mode"])), []).append(row)

    aggregates: List[Dict[str, object]] = []
    for (name, scenario, mode), values in grouped.items():
        pop = np.array([float(v["mean_pop_last50"]) for v in values])
        final_pop = np.array([float(v["final_pop"]) for v in values])
        extinct = np.array([float(v["extinct"]) for v in values])
        aggregates.append({
            "name": name,
            "scenario": scenario,
            "mode": mode,
            "mean_pop_last50": float(pop.mean()),
            "std_pop_last50": float(pop.std(ddof=1)) if len(pop) > 1 else 0.0,
            "mean_final_pop": float(final_pop.mean()),
            "survival_rate": float(1.0 - extinct.mean()),
        })

    # Normalise late population within each scenario/mode so hard and easy
    # scenarios contribute equally. Survival is already on [0, 1].
    strata: Dict[Tuple[str, str], List[Dict[str, object]]] = {}
    for row in aggregates:
        strata.setdefault((str(row["scenario"]), str(row["mode"])), []).append(row)
    for values in strata.values():
        pops = [float(v["mean_final_pop"]) for v in values]
        lo, hi = min(pops), max(pops)
        for value in values:
            norm_pop = 0.5 if hi == lo else (float(value["mean_final_pop"]) - lo) / (hi - lo)
            value["ecological_score"] = 0.50 * norm_pop + 0.50 * float(value["survival_rate"])

    by_candidate: Dict[str, List[Dict[str, object]]] = {}
    for row in aggregates:
        by_candidate.setdefault(str(row["name"]), []).append(row)

    summary: List[Dict[str, object]] = []
    baseline_score = float(np.mean([r["ecological_score"] for r in by_candidate["baseline"]]))
    for candidate in candidates():
        values = by_candidate[candidate.name]
        scores = np.array([float(v["ecological_score"]) for v in values])
        summary.append({
            **asdict(candidate),
            "mean_ecological_score": float(scores.mean()),
            "std_ecological_score": float(scores.std(ddof=1)) if len(scores) > 1 else 0.0,
            "delta_vs_baseline": float(scores.mean() - baseline_score),
            "mean_survival_rate": float(np.mean([v["survival_rate"] for v in values])),
            "mean_pop_last50": float(np.mean([v["mean_pop_last50"] for v in values])),
            "mean_final_pop": float(np.mean([v["mean_final_pop"] for v in values])),
            "n_strata": len(values),
        })
    summary.sort(key=lambda row: float(row["mean_ecological_score"]), reverse=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--replicas", type=int, default=10)
    parser.add_argument("--workers", type=int, default=min(8, mp.cpu_count()))
    parser.add_argument("--modes", nargs="+", choices=MODES, default=MODES)
    parser.add_argument("--append", action="store_true",
                        help="Keep existing rows for modes not selected in this run")
    args = parser.parse_args()

    tasks = [
        (candidate, scenario, mode, replica, args.steps)
        for candidate in candidates()
        for scenario in SCENARIOS
        for mode in args.modes
        for replica in range(args.replicas)
    ]
    print(f"Running {len(tasks)} episodes ({args.steps} steps, {args.workers} workers)...")
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers) as pool:
        rows = list(pool.imap_unordered(_worker, tasks, chunksize=1))

    detail_path = RESULTS_DIR / "online_learning_tuning_detail.csv"
    summary_path = RESULTS_DIR / "online_learning_tuning_summary.csv"
    if args.append and detail_path.exists():
        with detail_path.open(newline="", encoding="utf-8") as f:
            previous = list(csv.DictReader(f))
        # Replace, rather than duplicate, modes evaluated by this invocation.
        rows = [r for r in previous if str(r["mode"]) not in set(args.modes)] + rows
    _write_rows(detail_path, rows)
    summary = _summarise(rows)
    _write_rows(summary_path, summary)

    print("\nRanking (ecological score; higher is better):")
    for rank, row in enumerate(summary, 1):
        print(
            f"{rank:2d}. {row['name']:<18} score={row['mean_ecological_score']:.4f} "
            f"delta={row['delta_vs_baseline']:+.4f} survival={row['mean_survival_rate']:.3f}"
        )
    print(f"\nDetail:  {detail_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
