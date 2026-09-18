"""Fully parallelised automation: dataset generation + supervised training.

Workflow:
1. Dataset Generation (parallel) - RB Rand Evo top-5% x 5 scenarios
2. Supervised Training (parallel) - RB Rand Evo top-5% x 3 layer counts x 5 scenarios
"""

import io
import sys
import time
import torch
import torch.multiprocessing as mp
from typing import List

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

from config import SCENARIOS, setup_logging
import lib_training

console = Console()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
LAYERS_TO_TEST: List[int] = [1, 2, 3]
MAX_WORKERS = 10
TRAIN_MAX_STEPS = 2000
TRAIN_EPOCHS = 80
MIN_ROWS_DATASET = 100_000
MIN_EPISODES_DATASET = 10
MAX_ROWS_PER_EPISODE = 12_000

# Evolutionary top-5% dataset
EVO_TOP5_DATASET: str = "rb_rand_evo_top5"
EVO_TOP5_WEIGHT_TAG_SUFFIX: str = "sup_rb_evo_top5_auto"


# ---------------------------------------------------------------------------
# Worker functions (top-level for pickling)
# ---------------------------------------------------------------------------


def _dispatch_pool_call(call_item):
    """Run one (function, args) tuple inside pool workers."""
    func, args = call_item
    try:
        return func(*args)
    except Exception as exc:
        return False, f"Pool dispatch failure: {exc}", ""

def _capture_stdout():
    buf = io.StringIO()
    sys.stdout = buf
    return buf


def worker_gen_dataset(scenario: str, kind: str, best_alpha: float = None):
    buf = _capture_stdout()
    try:
        torch.set_num_threads(1)
        path = lib_training.generate_dataset(
            scenario,
            kind,
            best_alpha=best_alpha,
            min_rows=MIN_ROWS_DATASET,
            min_episodes=MIN_EPISODES_DATASET,
            max_rows_per_episode=MAX_ROWS_PER_EPISODE,
            min_survival=0,
            max_steps=TRAIN_MAX_STEPS,
        )
        return True, f"{scenario} ({kind})", buf.getvalue()
    except Exception as e:
        return False, f"{scenario} ({kind}): {e}", buf.getvalue()


def worker_train_evo_top5(scenario: str, n_layers: int, epochs: int):
    buf = _capture_stdout()
    try:
        torch.set_num_threads(1)
        _, move_acc, repro_acc = lib_training.train(
            scenario,
            n_layers,
            dataset_kind=EVO_TOP5_DATASET,
            epochs=epochs,
            output_weight_tag=EVO_TOP5_WEIGHT_TAG_SUFFIX,
        )
        return True, (
            f"{scenario} L{n_layers} ({EVO_TOP5_DATASET}) "
            f"move_acc={move_acc:.4f} repro_acc={repro_acc:.4f}"
        ), buf.getvalue()
    except Exception as e:
        return False, f"{scenario} L{n_layers} ({EVO_TOP5_DATASET}): {e}", buf.getvalue()


# ---------------------------------------------------------------------------
# Phase runners
# ---------------------------------------------------------------------------

def run_pool_task(context, func, args_list, phase_name):
    """Execute tasks with a Rich progress bar."""
    if not args_list:
        return

    total = len(args_list)
    ok_count = 0
    err_count = 0

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
        task_bar = progress.add_task(f"{phase_name}...", total=total)

        with context.Pool(processes=MAX_WORKERS) as pool:
            work_items = [(func, args) for args in args_list]
            for success, msg, log in pool.imap_unordered(
                _dispatch_pool_call,
                work_items,
                chunksize=1,
            ):
                try:
                    if success:
                        ok_count += 1
                        if log.strip():
                            for line in log.strip().splitlines():
                                progress.console.print(f"  [dim]{line.strip()}[/dim]")
                    else:
                        err_count += 1
                        progress.console.print(f"  [red][ERR][/red] {msg}")
                except Exception as e:
                    err_count += 1
                    progress.console.print(f"  [red][CRITICAL][/red] {e}")
                progress.advance(task_bar)

        progress.update(task_bar, description=f"[bold bright_green]{phase_name} done!")

    summary = Table(show_header=False, box=None, padding=(0, 2))
    summary.add_column(style="bold")
    summary.add_column()
    summary.add_row("[green]Completed[/green]", f"{ok_count}/{total}")
    if err_count > 0:
        summary.add_row("[red]Errors[/red]", str(err_count))
    console.print(summary)
    console.print()


def run_dataset_phase(ctx):
    """Generate supervised datasets in parallel (RB Rand Evo top-5%)."""
    console.print()
    console.print(
        Panel.fit(
            "[bold white]PHASE 1: DATASET GENERATION[/bold white]",
            border_style="blue",
            padding=(0, 4),
        )
    )
    # Evo-top5 does not use a fixed alpha: best_alpha=None
    tasks = [(scenario, EVO_TOP5_DATASET, None) for scenario in SCENARIOS]
    run_pool_task(ctx, worker_gen_dataset, tasks, "Dataset Generation")


def run_training_phase(ctx):
    """Run supervised training in parallel (rb_rand_evo_top5)."""
    console.print()
    console.print(
        Panel.fit(
            "[bold white]PHASE 2: SUPERVISED TRAINING[/bold white]",
            border_style="magenta",
            padding=(0, 4),
        )
    )

    tasks_evo_top5 = [
        (scenario, lay, TRAIN_EPOCHS)
        for scenario in SCENARIOS
        for lay in LAYERS_TO_TEST
    ]

    total_tasks = len(tasks_evo_top5)
    cfg_table = Table(show_header=False, box=None, padding=(0, 2))
    cfg_table.add_column(style="bold cyan")
    cfg_table.add_column(style="white")
    cfg_table.add_row("Scenarios", str(len(SCENARIOS)))
    cfg_table.add_row("Layers tested", ", ".join(f"L{l}" for l in LAYERS_TO_TEST))
    cfg_table.add_row("Dataset kinds", EVO_TOP5_DATASET)
    cfg_table.add_row("Minimum episodes", str(MIN_EPISODES_DATASET))
    cfg_table.add_row("Per-episode row cap", f"{MAX_ROWS_PER_EPISODE:,}")
    cfg_table.add_row("Training epochs", str(TRAIN_EPOCHS))
    cfg_table.add_row("Total tasks", str(total_tasks))
    console.print(cfg_table)
    console.print()

    run_pool_task(ctx, worker_train_evo_top5, tasks_evo_top5, "Training Supervised (rb_rand_evo_top5)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    try:
        ctx = mp.get_context("spawn")
    except ValueError:
        ctx = mp.get_context()

    console.print()
    console.print(
        Panel.fit(
            "[bold white]TRAINING AUTOMATION[/bold white]\n"
            f"[dim]Workers: {MAX_WORKERS} · Scenarios: {len(SCENARIOS)}[/dim]",
            border_style="bright_cyan",
            padding=(1, 4),
        )
    )

    console.print("\n  [bold]Choose mode:[/bold]")
    console.print("    [cyan]1)[/cyan] dataset      - Dataset generation (parallel)")
    console.print("    [cyan]2)[/cyan] training     - Supervised training (parallel)")
    console.print("    [cyan]3)[/cyan] all          - All (dataset + training)")

    choice = input("\n  Enter choice (1/2/3): ").strip()
    mode_map = {
        "1": "dataset",
        "2": "training",
        "3": "all",
    }
    mode = mode_map.get(choice)

    if not mode:
        console.print("[red]  Invalid choice.[/red]")
        sys.exit(1)

    t_start = time.time()

    if mode in {"dataset", "all", "full_dagger"}:
        run_dataset_phase(ctx)

    if mode in {"training", "all"}:
        run_training_phase(ctx)

    mins, secs = divmod(int(time.time() - t_start), 60)

    summary = Table.grid(padding=(0, 2))
    summary.add_column(style="bold")
    summary.add_column()
    summary.add_row("[cyan]Total time[/cyan]", f"{mins}m {secs}s")
    summary.add_row("[cyan]Mode[/cyan]", mode)

    console.print()
    console.print(Panel(
        summary,
        title="[bold bright_green]Automation completed[/bold bright_green]",
        border_style="bright_green",
        expand=False,
    ))
    console.print()


if __name__ == "__main__":
    setup_logging()
    mp.freeze_support()
    main()
