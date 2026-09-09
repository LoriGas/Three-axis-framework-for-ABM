"""Parameter-space exploration for survival analysis.

Generates one CSV:
  - explore_scenarios.csv : full (fr, bm) range → heatmap / Pareto frontier
"""

from __future__ import annotations

import csv
import argparse
import multiprocessing as mp
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

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

from config import RESULTS_DIR, get_params_for_scenario
from model import World
from simulation import _resolve_rb_variant

console = Console()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

NUM_SIMULATIONS: int = 500
MAX_STEPS: int = 2000
MAX_WORKERS: int = 10

REGEN_MIN, REGEN_MAX = 0.0, 1.5
METAB_MIN, METAB_MAX = 0.1, 3.1
GRID_STEPS: int = 8         # 8×8 regular grid
NUM_RANDOM: int = 1_000
OUTPUT_CSV: Path = RESULTS_DIR / "explore_scenarios.csv"

TARGETED_APPEND_POINTS: int = 100
TARGETED_APPEND_SIMULATIONS: int = 500
TARGETED_REGEN_MIN, TARGETED_REGEN_MAX = 0.0, 0.3
TARGETED_METAB_MIN, TARGETED_METAB_MAX = 0.1, 1.25
TARGETED_RANDOM_SEED: int = 20260521
TARGETED_SEED_BASE: int = 900_000
TARGETED_SURVIVAL_LEVELS: Tuple[float, ...] = (
    0.05, 0.15, 0.25, 0.40, 0.50, 0.60, 0.75, 0.85, 0.95,
)

CSV_FIELDS: List[str] = [
    "regen",
    "metab",
    "survival_probability",
    "mean_final_population",
    "std_final_population",
    "mean_births",
    "std_births",
    "has_reproduction",
]


# ---------------------------------------------------------------------------
# Worker: simulate one point
# ---------------------------------------------------------------------------

def _simulate_point(args: Tuple[float, float, int] | Tuple[float, float, int, int]) -> Dict[str, float]:
    if len(args) == 3:
        regen, metab, seed_base = args
        num_simulations = NUM_SIMULATIONS
    else:
        regen, metab, seed_base, num_simulations = args

    fixed_alpha, fixed_beta, mutate_propensity, _, _ = _resolve_rb_variant("rb_rand_evo")

    final_pops: List[int] = []
    total_births: List[int] = []
    survivors = 0

    for i in range(num_simulations):
        seed = seed_base + i
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        params = get_params_for_scenario("S3")
        params.food_regen = regen
        params.base_metabolism = metab
        params.initial_alpha = fixed_alpha if fixed_alpha is not None else -1.0
        params.initial_beta = fixed_beta if fixed_beta is not None else -1.0

        world = World(params, agent_type="rule", mutate_propensity=mutate_propensity)

        births_this_run = 0
        for _ in range(MAX_STEPS):
            world.step()
            births_this_run += world.births_step
            if not world.agents:
                break

        final_pop = len(world.agents)
        final_pops.append(final_pop)
        total_births.append(births_this_run)
        if final_pop > 0:
            survivors += 1

    births_mean = float(np.mean(total_births))
    return {
        "regen": float(regen),
        "metab": float(metab),
        "survival_probability": survivors / num_simulations,
        "mean_final_population": float(np.mean(final_pops)),
        "std_final_population": float(np.std(final_pops)),
        "mean_births": births_mean,
        "std_births": float(np.std(total_births)),
        "has_reproduction": 1 if births_mean > 1.0 else 0,
    }


# ---------------------------------------------------------------------------
# Point generator
# ---------------------------------------------------------------------------

def generate_points() -> List[Tuple[float, float, int]]:
    """8×8 regular grid + 800 random points covering the full (fr, bm) range."""
    points: List[Tuple[float, float, int]] = []
    seed_counter = 42_000

    for regen in np.linspace(REGEN_MIN, REGEN_MAX, GRID_STEPS):
        for metab in np.linspace(METAB_MIN, METAB_MAX, GRID_STEPS):
            points.append((round(float(regen), 3), round(float(metab), 3), seed_counter))
            seed_counter += NUM_SIMULATIONS

    rng = np.random.RandomState(123)
    for _ in range(NUM_RANDOM):
        regen = round(float(rng.uniform(REGEN_MIN, REGEN_MAX)), 3)
        metab = round(float(rng.uniform(METAB_MIN, METAB_MAX)), 3)
        points.append((regen, metab, seed_counter))
        seed_counter += NUM_SIMULATIONS

    return points


def _load_existing_points(path: Path) -> set[Tuple[float, float]]:
    if not path.exists():
        return set()

    existing: set[Tuple[float, float]] = set()
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                existing.add((round(float(row["regen"]), 3), round(float(row["metab"]), 3)))
            except (KeyError, TypeError, ValueError):
                continue
    return existing


def _load_targeted_observations(path: Path) -> List[Tuple[float, float, float]]:
    if not path.exists():
        return []

    observations: List[Tuple[float, float, float]] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                regen = float(row["regen"])
                metab = float(row["metab"])
                survival = float(row["survival_probability"])
            except (KeyError, TypeError, ValueError):
                continue
            if (
                TARGETED_REGEN_MIN <= regen < TARGETED_REGEN_MAX
                and TARGETED_METAB_MIN <= metab < TARGETED_METAB_MAX
            ):
                observations.append((regen, metab, survival))
    return observations


def _normalised_distance_sq(
    regen_a: float,
    metab_a: float,
    regen_b: float,
    metab_b: float,
) -> float:
    regen_span = TARGETED_REGEN_MAX - TARGETED_REGEN_MIN
    metab_span = TARGETED_METAB_MAX - TARGETED_METAB_MIN
    dx = (regen_a - regen_b) / max(regen_span, 1e-9)
    dy = (metab_a - metab_b) / max(metab_span, 1e-9)
    return dx * dx + dy * dy


def _predict_survival_knn(
    regen: float,
    metab: float,
    observations: List[Tuple[float, float, float]],
    k: int = 12,
) -> Tuple[float, float]:
    neighbours = sorted(
        (
            _normalised_distance_sq(regen, metab, obs_regen, obs_metab),
            survival,
        )
        for obs_regen, obs_metab, survival in observations
    )[: max(1, min(k, len(observations)))]

    min_dist = float(np.sqrt(neighbours[0][0]))
    weights = [1.0 / (dist_sq + 1e-6) for dist_sq, _ in neighbours]
    total_weight = sum(weights)
    prediction = sum(w * survival for w, (_, survival) in zip(weights, neighbours)) / total_weight
    return float(prediction), min_dist


def _make_frontier_candidate_pool(
    rng: np.random.RandomState,
    observations: List[Tuple[float, float, float]],
    existing_points: set[Tuple[float, float]],
    n_points: int,
) -> List[Dict[str, float]]:
    pool_size = max(20_000, n_points * 250)
    candidates: List[Dict[str, float]] = []
    seen = set(existing_points)

    for _ in range(pool_size):
        regen = round(float(rng.uniform(TARGETED_REGEN_MIN, TARGETED_REGEN_MAX)), 3)
        metab = round(float(rng.uniform(TARGETED_METAB_MIN, TARGETED_METAB_MAX)), 3)
        point = (regen, metab)
        if point in seen:
            continue
        seen.add(point)

        survival_hat, dist_existing = _predict_survival_knn(regen, metab, observations)
        nearest_level_error = min(abs(survival_hat - level) for level in TARGETED_SURVIVAL_LEVELS)
        candidates.append(
            {
                "regen": regen,
                "metab": metab,
                "survival_hat": survival_hat,
                "dist_existing": dist_existing,
                "level_error": nearest_level_error,
            }
        )

    return candidates


def _generate_frontier_focused_points(
    n_points: int,
    num_simulations: int,
    existing_points: set[Tuple[float, float]],
    rng: np.random.RandomState,
) -> List[Tuple[float, float, int, int]]:
    observations = _load_targeted_observations(OUTPUT_CSV)
    if len(observations) < 12:
        return []

    candidates = _make_frontier_candidate_pool(rng, observations, existing_points, n_points)
    if not candidates:
        return []

    selected: List[Dict[str, float]] = []
    used = set(existing_points)
    seed_counter = TARGETED_SEED_BASE
    per_level = int(np.ceil(n_points / len(TARGETED_SURVIVAL_LEVELS)))

    for level in TARGETED_SURVIVAL_LEVELS:
        ranked = sorted(
            candidates,
            key=lambda c: (
                abs(c["survival_hat"] - level),
                -c["dist_existing"],
            ),
        )

        picked_for_level = 0
        min_sep = 0.030
        while picked_for_level < per_level and len(selected) < n_points:
            added = False
            for cand in ranked:
                point = (cand["regen"], cand["metab"])
                if point in used:
                    continue
                if selected:
                    nearest_selected = min(
                        np.sqrt(
                            _normalised_distance_sq(
                                cand["regen"],
                                cand["metab"],
                                prev["regen"],
                                prev["metab"],
                            )
                        )
                        for prev in selected
                    )
                    if nearest_selected < min_sep:
                        continue
                selected.append(cand)
                used.add(point)
                picked_for_level += 1
                added = True
                break
            if not added:
                min_sep *= 0.75
                if min_sep < 0.005:
                    break

    if len(selected) < n_points:
        ranked = sorted(
            candidates,
            key=lambda c: (
                c["level_error"],
                -c["dist_existing"],
            ),
        )
        for cand in ranked:
            point = (cand["regen"], cand["metab"])
            if point in used:
                continue
            selected.append(cand)
            used.add(point)
            if len(selected) == n_points:
                break

    points: List[Tuple[float, float, int, int]] = []
    for cand in selected[:n_points]:
        points.append((cand["regen"], cand["metab"], seed_counter, num_simulations))
        seed_counter += num_simulations
    return points


def generate_targeted_points(
    n_points: int,
    num_simulations: int,
    *,
    existing_points: set[Tuple[float, float]] | None = None,
    random_seed: int = TARGETED_RANDOM_SEED,
) -> List[Tuple[float, float, int, int]]:
    """Frontier-focused points in the low-regeneration / low-metabolism area."""
    existing_points = existing_points or set()
    rng = np.random.RandomState(random_seed)
    frontier_points = _generate_frontier_focused_points(
        n_points,
        num_simulations,
        existing_points,
        rng,
    )
    if len(frontier_points) == n_points:
        return frontier_points

    points: List[Tuple[float, float, int, int]] = []
    used = set(existing_points)
    seed_counter = TARGETED_SEED_BASE

    while len(points) < n_points:
        batch_size = max(n_points - len(points), 64)
        regen_bins = rng.permutation(batch_size)
        metab_bins = rng.permutation(batch_size)
        regen_values = (
            TARGETED_REGEN_MIN
            + (regen_bins + rng.rand(batch_size)) / batch_size
            * (TARGETED_REGEN_MAX - TARGETED_REGEN_MIN)
        )
        metab_values = (
            TARGETED_METAB_MIN
            + (metab_bins + rng.rand(batch_size)) / batch_size
            * (TARGETED_METAB_MAX - TARGETED_METAB_MIN)
        )

        for regen, metab in zip(regen_values, metab_values):
            point = (round(float(regen), 3), round(float(metab), 3))
            if point in used:
                continue
            used.add(point)
            points.append((point[0], point[1], seed_counter, num_simulations))
            seed_counter += num_simulations
            if len(points) == n_points:
                break

    return points


# ---------------------------------------------------------------------------
# Run sweep
# ---------------------------------------------------------------------------

def run_sweep(points: List[Tuple[float, float, int]]) -> None:
    total = len(points)

    cfg_table = Table(show_header=False, box=None, padding=(0, 2))
    cfg_table.add_column(style="bold cyan")
    cfg_table.add_column(style="white")
    cfg_table.add_row("Points to explore", str(total))
    cfg_table.add_row("Simulations per point", str(NUM_SIMULATIONS))
    cfg_table.add_row("Total simulations", f"{total * NUM_SIMULATIONS:,}")
    cfg_table.add_row("Workers", str(MAX_WORKERS))
    cfg_table.add_row("food_regen range", f"[{REGEN_MIN}, {REGEN_MAX}]")
    cfg_table.add_row("base_metabolism range", f"[{METAB_MIN}, {METAB_MAX}]")
    cfg_table.add_row("Output CSV", str(OUTPUT_CSV))
    console.print(Panel(cfg_table, title="[bold]Full-range sweep (Pareto heatmap)[/bold]", border_style="dim", expand=False))

    results: List[Dict[str, float]] = []
    ctx = mp.get_context("spawn")

    console.print()
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
        task_id = progress.add_task("Sweep in progress...", total=total)

        with ctx.Pool(processes=MAX_WORKERS) as pool:
            for result in pool.imap_unordered(_simulate_point, points):
                results.append(result)
                progress.advance(task_id)

        progress.update(task_id, description="[bold bright_green]Sweep completed!")

    if not results:
        raise RuntimeError("No results produced by the sweep.")

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(results)

    survivors_over_half = sum(1 for row in results if row["survival_probability"] > 0.5)
    reproductive_points = sum(1 for row in results if row["has_reproduction"])

    stat_table = Table(show_header=False, box=None, padding=(0, 2))
    stat_table.add_column(style="bold")
    stat_table.add_column()
    stat_table.add_row("[green]Points with survival > 50%[/green]", f"{survivors_over_half}/{total}")
    stat_table.add_row("[green]Points with reproduction[/green]", f"{reproductive_points}/{total}")
    stat_table.add_row("[blue]CSV saved[/blue]", str(OUTPUT_CSV))
    console.print(Panel(stat_table, title="[bold]Statistics[/bold]", border_style="dim", expand=False))
    console.print()


def append_targeted_sweep(points: List[Tuple[float, float, int, int]]) -> None:
    total = len(points)
    if not points:
        console.print("[yellow]No new targeted points to append.[/yellow]")
        return

    num_simulations = points[0][3]

    cfg_table = Table(show_header=False, box=None, padding=(0, 2))
    cfg_table.add_column(style="bold cyan")
    cfg_table.add_column(style="white")
    cfg_table.add_row("New points", str(total))
    cfg_table.add_row("Simulations per point", str(num_simulations))
    cfg_table.add_row("Total simulations", f"{total * num_simulations:,}")
    cfg_table.add_row("Workers", str(MAX_WORKERS))
    cfg_table.add_row("Sampling", "frontier-focused KNN from existing CSV")
    cfg_table.add_row("food_regen range", f"[{TARGETED_REGEN_MIN}, {TARGETED_REGEN_MAX})")
    cfg_table.add_row("base_metabolism range", f"[{TARGETED_METAB_MIN}, {TARGETED_METAB_MAX})")
    cfg_table.add_row("Output CSV", str(OUTPUT_CSV))
    console.print(Panel(cfg_table, title="[bold]Targeted low-fr / low-bm append[/bold]", border_style="dim", expand=False))

    results: List[Dict[str, float]] = []
    ctx = mp.get_context("spawn")

    console.print()
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
        task_id = progress.add_task("Targeted append in progress...", total=total)

        with ctx.Pool(processes=MAX_WORKERS) as pool:
            for result in pool.imap_unordered(_simulate_point, points):
                results.append(result)
                progress.advance(task_id)

        progress.update(task_id, description="[bold bright_green]Targeted append completed!")

    write_header = not OUTPUT_CSV.exists() or OUTPUT_CSV.stat().st_size == 0
    with open(OUTPUT_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerows(results)

    stat_table = Table(show_header=False, box=None, padding=(0, 2))
    stat_table.add_column(style="bold")
    stat_table.add_column()
    stat_table.add_row("[green]Rows appended[/green]", str(len(results)))
    stat_table.add_row("[blue]CSV updated[/blue]", str(OUTPUT_CSV))
    console.print(Panel(stat_table, title="[bold]Targeted append statistics[/bold]", border_style="dim", expand=False))
    console.print()


def choose_run_mode() -> str:
    console.print()
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="bold yellow", justify="right")
    table.add_column(style="white")
    table.add_row("1", "Full sweep: rebuild explore_scenarios.csv from scratch")
    table.add_row(
        "2",
        f"Append frontier-focused points only: add {TARGETED_APPEND_POINTS} points with fr < 0.3 and bm < 1.25",
    )
    console.print(Panel(table, title="[bold]Choose exploration mode[/bold]", border_style="dim", expand=False))

    while True:
        answer = input("  > Choice [1/2]: ").strip()
        if answer == "1":
            return "full"
        if answer == "2":
            return "append_targeted"
        print("Invalid choice. Type 1 or 2.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Explore scenario parameter space.")
    parser.add_argument(
        "--append-targeted",
        action="store_true",
        help="Append extra points in the low-regeneration / low-metabolism region.",
    )
    parser.add_argument(
        "--full-sweep",
        action="store_true",
        help="Run the full sweep and overwrite the CSV.",
    )
    parser.add_argument("--targeted-points", type=int, default=TARGETED_APPEND_POINTS)
    parser.add_argument("--targeted-simulations", type=int, default=TARGETED_APPEND_SIMULATIONS)
    parser.add_argument("--targeted-seed", type=int, default=TARGETED_RANDOM_SEED)
    args = parser.parse_args()

    if args.append_targeted and args.full_sweep:
        parser.error("Choose only one mode: --append-targeted or --full-sweep.")

    if args.append_targeted:
        run_mode = "append_targeted"
    elif args.full_sweep:
        run_mode = "full"
    elif len(sys.argv) == 1:
        run_mode = choose_run_mode()
    else:
        run_mode = "full"

    t_start = time.perf_counter()

    console.print()
    console.print(
        Panel.fit(
            "[bold white]PARAMETER SPACE EXPLORATION[/bold white]\n"
            "[dim]Survival map in the (food_regen, base_metabolism) plane[/dim]",
            border_style="bright_cyan",
            padding=(1, 4),
        )
    )

    if run_mode == "append_targeted":
        existing = _load_existing_points(OUTPUT_CSV)
        points = generate_targeted_points(
            args.targeted_points,
            args.targeted_simulations,
            existing_points=existing,
            random_seed=args.targeted_seed,
        )
        append_targeted_sweep(points)
    else:
        run_sweep(generate_points())

    elapsed = time.perf_counter() - t_start
    minutes, seconds = divmod(int(elapsed), 60)
    console.print(
        Panel.fit(
            f"[bold bright_green]Done in {minutes}m {seconds}s[/bold bright_green]",
            border_style="bright_green",
        )
    )
    console.print()


if __name__ == "__main__":
    mp.freeze_support()
    main()
