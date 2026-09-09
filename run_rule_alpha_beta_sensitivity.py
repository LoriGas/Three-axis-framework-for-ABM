"""Two-dimensional sensitivity of fixed rule agents to alpha and beta.

The default design evaluates a 5 x 5 grid in the discriminating scenarios
S2--S4, with 300 independent 2,000-step episodes per cell. Completed cells are
saved immediately, so an interrupted run can resume without repeating them.

Outputs:
- results/rule_alpha_beta_sensitivity_summary.csv
- results/rule_alpha_beta_sensitivity_replicates.csv
"""

from __future__ import annotations

import csv
import math
import multiprocessing as mp
import os
import time
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import torch
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn

from config import RESULTS_DIR, get_params_for_scenario, setup_logging
from run_analysis import MAX_STEPS, _run_episode


console = Console()

SCENARIOS: List[str] = ["S2", "S3", "S4"]
ALPHAS: List[float] = [0.0, 0.5, 1.0, 1.5, 2.0]
BETAS: List[float] = [0.0, 0.25, 0.5, 0.75, 1.0]
NUM_EPISODES: int = 300
MAX_WORKERS: int = 10

SUMMARY_CSV: Path = RESULTS_DIR / "rule_alpha_beta_sensitivity_summary.csv"
REPLICATE_CSV: Path = RESULTS_DIR / "rule_alpha_beta_sensitivity_replicates.csv"

SUMMARY_HEADER = [
    "scenario", "alpha", "beta", "n_episodes", "n_extinctions",
    "extinction_rate", "wilson_low", "wilson_high",
    "avg_final_population", "std_final_population",
    "avg_final_population_survivors",
]
REPLICATE_HEADER = [
    "scenario", "alpha", "beta", "replica", "seed", "extinct",
    "pop_final", "n_steps_alive", "reproduction_events",
    "movement_frequency", "avg_harvested_food",
]


def _key(row: Sequence[object]) -> Tuple[str, float, float]:
    return str(row[0]), round(float(row[1]), 8), round(float(row[2]), 8)


def _load_rows(path: Path) -> List[List[str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with open(path, newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        return [row for row in reader if row]


def _atomic_write(path: Path, header: Sequence[str], rows: Iterable[Sequence[object]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _wilson(extinctions: int, episodes: int, z: float = 1.96) -> Tuple[float, float]:
    estimate = extinctions / episodes
    denominator = 1.0 + z * z / episodes
    centre = (estimate + z * z / (2.0 * episodes)) / denominator
    half_width = z * math.sqrt(
        estimate * (1.0 - estimate) / episodes + z * z / (4.0 * episodes * episodes)
    ) / denominator
    return max(0.0, centre - half_width), min(1.0, centre + half_width)


def _worker(task: Tuple[str, float, float]) -> Tuple[List[object], List[List[object]]]:
    torch.set_num_threads(1)
    scenario, alpha, beta = task
    params = get_params_for_scenario(scenario)
    scenario_offset = SCENARIOS.index(scenario) * 100_000
    rows: List[List[object]] = []
    final_populations: List[int] = []
    surviving_populations: List[int] = []
    extinctions = 0

    for replica in range(NUM_EPISODES):
        # The same scenario/replica seed is used throughout the grid. This is a
        # common-random-number design; trajectories can still diverge after the
        # first parameter-dependent action.
        seed = 3_000_000 + scenario_offset + replica
        result = _run_episode(
            params,
            0,
            mode="rule",
            fixed_alpha=alpha,
            fixed_beta=beta,
            mutate_propensity=False,
            random_offspring_propensity=False,
            propensity_learning=False,
            net_learning=False,
            seed=seed,
        )
        extinctions += result.extinct
        final_populations.append(result.pop_final)
        if not result.extinct:
            surviving_populations.append(result.pop_final)
        rows.append([
            scenario, alpha, beta, replica, seed, result.extinct,
            result.pop_final, result.n_steps_alive, result.reproduction_events,
            result.movement_frequency, result.avg_harvested_food,
        ])

    low, high = _wilson(extinctions, NUM_EPISODES)
    summary = [
        scenario,
        alpha,
        beta,
        NUM_EPISODES,
        extinctions,
        extinctions / NUM_EPISODES,
        low,
        high,
        float(np.mean(final_populations)),
        float(np.std(final_populations, ddof=1)),
        float(np.mean(surviving_populations)) if surviving_populations else float("nan"),
    ]
    return summary, rows


def _parse_values(raw: str, allowed: Sequence[object]) -> List[object]:
    if not raw.strip() or raw.strip().lower() in {"all", "*"}:
        return list(allowed)
    lookup = {str(value): value for value in allowed}
    requested = [part.strip() for part in raw.split(",") if part.strip()]
    invalid = [value for value in requested if value not in lookup]
    if invalid:
        raise ValueError(", ".join(invalid))
    return list(dict.fromkeys(lookup[value] for value in requested))


def _choose_tasks(all_tasks: List[Tuple[str, float, float]], completed: set) -> Tuple[List[Tuple[str, float, float]], bool]:
    if SUMMARY_CSV.exists():
        while True:
            answer = input(
                "Resume missing cells, replace a selection, or rewrite everything? [r/s/a] "
            ).strip().lower()
            if answer in {"", "r", "resume"}:
                return [task for task in all_tasks if _key(task) not in completed], True
            if answer in {"a", "all", "rewrite"}:
                return all_tasks, False
            if answer in {"s", "select"}:
                break
            console.print("[yellow]Use 'r', 's', or 'a'.[/yellow]")
    else:
        while True:
            answer = input(
                "Run the full grid or choose scenarios/values? [a/s] "
            ).strip().lower()
            if answer in {"", "a", "all"}:
                return all_tasks, False
            if answer in {"s", "select"}:
                break
            console.print("[yellow]Use 'a' or 's'.[/yellow]")

    while True:
        try:
            scenarios = _parse_values(input(f"Scenarios ({', '.join(SCENARIOS)}): "), SCENARIOS)
            alphas = _parse_values(input(f"Alpha ({', '.join(map(str, ALPHAS))}): "), ALPHAS)
            betas = _parse_values(input(f"Beta ({', '.join(map(str, BETAS))}): "), BETAS)
            break
        except ValueError as exc:
            console.print(f"[yellow]Invalid values: {exc}. Try again.[/yellow]")

    selected = [
        task for task in all_tasks
        if task[0] in scenarios and task[1] in alphas and task[2] in betas
    ]
    return selected, True


def main() -> None:
    started = time.perf_counter()
    all_tasks = [(scenario, alpha, beta) for scenario in SCENARIOS for alpha in ALPHAS for beta in BETAS]
    summary_rows = _load_rows(SUMMARY_CSV)
    replicate_rows = _load_rows(REPLICATE_CSV)
    completed = {_key(row) for row in summary_rows}
    tasks, preserve_existing = _choose_tasks(all_tasks, completed)
    if not preserve_existing:
        summary_rows = []
        replicate_rows = []

    console.print(
        f"[bold]Alpha-beta sensitivity:[/bold] {len(tasks)} cells, "
        f"{NUM_EPISODES} replicates per cell "
        f"([yellow]{len(tasks) * NUM_EPISODES:,} episodes[/yellow]), "
        f"{MAX_STEPS} maximum steps."
    )
    if not tasks:
        console.print("[green]All requested cells are already complete.[/green]")
        return

    ctx = mp.get_context("spawn")
    try:
        with Progress(
            TextColumn("[bold]{task.description}"),
            BarColumn(bar_width=40),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            progress_id = progress.add_task("Simulations in progress", total=len(tasks))
            with ctx.Pool(processes=MAX_WORKERS) as pool:
                for summary, rows in pool.imap_unordered(_worker, tasks):
                    completed_key = _key(summary)
                    summary_rows = [row for row in summary_rows if _key(row) != completed_key]
                    replicate_rows = [row for row in replicate_rows if _key(row) != completed_key]
                    summary_rows.append(summary)
                    replicate_rows.extend(rows)
                    summary_rows.sort(key=_key)
                    replicate_rows.sort(key=lambda row: (*_key(row), int(float(row[3]))))
                    _atomic_write(SUMMARY_CSV, SUMMARY_HEADER, summary_rows)
                    _atomic_write(REPLICATE_CSV, REPLICATE_HEADER, replicate_rows)
                    scenario, alpha, beta = completed_key
                    progress.console.print(
                        f"[green]✓ Saved[/green] {scenario} · alpha={alpha:g} · beta={beta:g}"
                    )
                    progress.advance(progress_id)
    except KeyboardInterrupt:
        console.print(
            "\n[yellow]Interrupted. All cells marked as saved remain available; "
            "choose 'r' on the next run to resume.[/yellow]"
        )
        return

    elapsed = time.perf_counter() - started
    console.print(
        f"[bold green]Analysis complete.[/bold green] "
        f"{len(tasks)} cells in {elapsed / 60.0:.1f} minutes."
    )
    console.print(f"Results: {SUMMARY_CSV}")


if __name__ == "__main__":
    mp.freeze_support()
    setup_logging()
    main()
