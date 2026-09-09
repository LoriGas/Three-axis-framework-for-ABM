"""Entry point: interactive menu and argparse for GUI, batch, and gene-sweep modes."""

import argparse
import multiprocessing
import sys
from typing import Dict, List, Optional, Tuple

from config import (
    LLM_VARIANT_TAGS,
    MLP_OFFLINE_INIT,
    MLP_VARIANT_TAGS,
    RULE_VARIANT_TAGS,
    VALID_SCENARIOS,
)
from gui import run_gui
from simulation import run_batch, run_rb_gene_sweep_batch

# ---------------------------------------------------------------------------
# Variant metadata for the interactive menu
# ---------------------------------------------------------------------------

TRAINING_DATASETS: List[Tuple[str, str]] = [
    ("sup_rb_evo_top5_auto",            "BC — RB evolutionary top-5% agents (run_automation)"),
    ("sup_rb_evo_top5_auto_uncertainty", "BC — RB top-5% with perception uncertainty"),
    ("sup_rb_fix_fix_auto",             "BC — RB fixed alpha + beta (run_automation)"),
    ("sup_rb_fix_fix_auto_dagger_it3",  "DAgger (3 iter) — RB fixed alpha + beta (run_automation)"),
]

MLP_VARIANTS: List[Tuple[str, str]] = [
    ("mlp_rand_fix",     "MLP | Random init + Fixed          (V8)"),
    ("mlp_rand_rand",    "MLP | Random init + Random offspring (V9)"),
    ("mlp_rand_evo",     "MLP | Random init + Evolution       (V10)"),
    ("mlp_rand_learn",   "MLP | Random init + Online learning (V11)"),
    ("mlp_offline_fix",  "MLP | Offline BC + Fixed            (V12)"),
    ("mlp_offline_evo",  "MLP | Offline BC + Evolution        (V13)"),
    ("mlp_offline_learn","MLP | Offline BC + Online learning  (V14)"),
]

LLM_VARIANTS: List[Tuple[str, str]] = [
    ("llm", "LLM | Chat-completions policy with local fallback"),
]

RULE_VARIANTS: List[Tuple[str, str]] = [
    ("rb_rand_fix",   "Rule-Based | Random alpha/beta + Fixed inheritance (V1)"),
    ("rb_rand_rand",  "Rule-Based | Random alpha + Random offspring (V2)"),
    ("rb_rand_evo",   "Rule-Based | Random alpha + Evolution        (V3)"),
    ("rb_rand_learn", "Rule-Based | Random alpha + Online learning  (V4)"),
    ("rb_fix_fix",    "Rule-Based | Fixed alpha=1 + Fixed           (V5)"),
    ("rb_fix_evo",    "Rule-Based | Fixed alpha=1 + Evolution       (V6)"),
    ("rb_fix_learn",  "Rule-Based | Fixed alpha=1 + Online learning (V7)"),
]

SCENARIO_DESC: Dict[str, str] = {
    "S1": "Very high metabolism / Very low regeneration",
    "S2": "High metabolism / Medium-low regeneration",
    "S3": "Medium metabolism / Medium regeneration",
    "S4": "Medium-low metabolism / Medium-high regeneration",
    "S5": "Low metabolism / High regeneration",
}

# ---------------------------------------------------------------------------
# Terminal utilities
# ---------------------------------------------------------------------------

def _ask_choice(prompt: str, options: List[str]) -> str:
    while True:
        answer = input(prompt).strip()
        if answer in options:
            return answer
        print("Invalid choice!")


def _print_header(title: str) -> None:
    w = 54
    print("\n" + "\033[36m" + "╔" + "═" * w + "╗")
    print("║" + title.center(w) + "║")
    print("╚" + "═" * w + "╝" + "\033[0m")


def _print_section(title: str) -> None:
    print(f"\n\033[1;37m  {title}\033[0m")
    print("  " + "─" * 40)


def _print_option(num: int, label: str, desc: str = "") -> None:
    desc_str = f"  \033[2m{desc}\033[0m" if desc else ""
    print(f"  \033[1;33m{num}\033[0m. \033[37m{label}\033[0m{desc_str}")


# ---------------------------------------------------------------------------
# Interactive menu
# ---------------------------------------------------------------------------

def interactive_menu() -> Tuple[str, str, str, int, int, Optional[int], str, str, float]:
    """Display a menu to choose execution mode and parameters."""
    _print_header("AGENT SIMULATION")

    _print_section("Execution mode")
    _print_option(1, "GUI", "Interactive graphical visualisation")
    _print_option(2, "BATCH", "Mass simulation with grid search")
    exec_mode = "gui" if _ask_choice("\n  \033[36m>\033[0m Choice: ", ["1", "2"]) == "1" else "batch"

    _print_section("Scenario")
    for i, s in enumerate(VALID_SCENARIOS, 1):
        _print_option(i, s, SCENARIO_DESC.get(s, ""))
    idx = int(_ask_choice(
        f"\n  \033[36m>\033[0m Choice (1-{len(VALID_SCENARIOS)}): ",
        [str(i) for i in range(1, len(VALID_SCENARIOS) + 1)],
    ))
    scenario = VALID_SCENARIOS[idx - 1]

    _print_section("Agent type")
    _print_option(1, "Rule-based", "Fixed decision rule with genes alpha and beta")
    _print_option(2, "MLP", "Feed-forward neural network")
    _print_option(3, "Random", "Completely random actions")
    _print_option(4, "LLM", "Language-model policy with optional API calls")
    agent_type = _ask_choice("\n  \033[36m>\033[0m Choice: ", ["1", "2", "3", "4"])

    rule_variant = "rb_fix_fix"
    agent = "rule"
    if agent_type == "1":
        _print_section("Rule-based variant")
        for i, (tag, desc) in enumerate(RULE_VARIANTS, 1):
            _print_option(i, desc, tag)
        v_idx = int(_ask_choice(
            f"\n  \033[36m>\033[0m Choice (1-{len(RULE_VARIANTS)}): ",
            [str(i) for i in range(1, len(RULE_VARIANTS) + 1)],
        ))
        rule_variant = RULE_VARIANTS[v_idx - 1][0]
    elif agent_type == "2":
        _print_section("MLP variant")
        for i, (tag, desc) in enumerate(MLP_VARIANTS, 1):
            _print_option(i, desc, tag)
        m_idx = int(_ask_choice(
            f"\n  \033[36m>\033[0m Choice (1-{len(MLP_VARIANTS)}): ",
            [str(i) for i in range(1, len(MLP_VARIANTS) + 1)],
        ))
        agent = MLP_VARIANTS[m_idx - 1][0]
    elif agent_type == "4":
        _print_section("LLM variant")
        for i, (tag, desc) in enumerate(LLM_VARIANTS, 1):
            _print_option(i, desc, tag)
        l_idx = int(_ask_choice(
            f"\n  \033[36m>\033[0m Choice (1-{len(LLM_VARIANTS)}): ",
            [str(i) for i in range(1, len(LLM_VARIANTS) + 1)],
        ))
        agent = LLM_VARIANTS[l_idx - 1][0]
    else:
        agent = "random"

    weight_tag = TRAINING_DATASETS[0][0]
    if agent in MLP_OFFLINE_INIT:
        _print_section("Training weights")
        for i, (tag, desc) in enumerate(TRAINING_DATASETS, 1):
            _print_option(i, desc, tag)
        w_idx = int(_ask_choice(
            f"\n  \033[36m>\033[0m Choice (1-{len(TRAINING_DATASETS)}): ",
            [str(i) for i in range(1, len(TRAINING_DATASETS) + 1)],
        ))
        weight_tag = TRAINING_DATASETS[w_idx - 1][0]

    _print_section("Parameters")
    raw = input("  Steps \033[2m(default 500, -1 = infinite)\033[0m: ").strip()
    steps = int(raw) if raw else 500

    raw = input("  Perception noise sigma \033[2m(default 0; uncertainty mode 0.5)\033[0m: ").strip()
    obs_noise_sigma = max(0.0, float(raw)) if raw else 0.0

    layers = 1
    if agent not in ("rule", "random", "llm"):
        raw = input("  Hidden layers \033[2m(default 1)\033[0m: ").strip()
        layers = int(raw) if raw else 1

    replicas: Optional[int] = None
    if exec_mode == "batch":
        raw = input("  Replicas per config \033[2m(default 5)\033[0m: ").strip()
        replicas = int(raw) if raw else 5

    print(f"\n  \033[36m{'─' * 40}\033[0m")
    print(f"  \033[1mSummary:\033[0m {exec_mode.upper()} | {scenario} | {agent}")
    if agent == "rule":
        print(f"  RB variant: {rule_variant}")
    if agent in MLP_OFFLINE_INIT:
        print(f"  Training dataset: {weight_tag}")
    summary = f"  Steps: {steps}"
    if agent not in ("rule", "random", "llm"):
        summary += f" | Layers: {layers}"
    print(summary, end="")
    print(f" | Obs noise σ: {obs_noise_sigma:.2f}", end="")
    if replicas is not None:
        print(f" | Replicas: {replicas}", end="")
    print()

    return exec_mode, scenario, agent, steps, layers, replicas, weight_tag, rule_variant, obs_noise_sigma


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    multiprocessing.freeze_support()

    parser = argparse.ArgumentParser(description="Agent simulation")
    parser.add_argument("--scenario", type=str, default=None,
                        help=f"One of: {', '.join(VALID_SCENARIOS)}; or 'all' for gene_sweep on all")
    parser.add_argument("--mode", type=str, default=None,
                        help="gui | batch | gene_sweep")
    parser.add_argument("--agent", type=str, default=None,
                        help="rule | random | " + " | ".join(MLP_VARIANT_TAGS + LLM_VARIANT_TAGS))
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--replicas", type=int, default=5)
    parser.add_argument("--weights", type=str, default="sup_rb_evo_top5_auto")
    parser.add_argument("--variant", type=str, default="rb_fix_fix",
                        help="Rule-based variant: " + " | ".join(RULE_VARIANT_TAGS))
    parser.add_argument("--obs-noise-sigma", type=float, default=0.0,
                        help="Gaussian sigma for uncertainty in neighbouring-food perception")

    args = parser.parse_args()

    needs_menu = args.mode is None or (
        args.mode != "gene_sweep" and (args.scenario is None or args.agent is None)
    )

    if needs_menu:
        print("\n[INFO] Starting interactive mode...")
        mode, scenario, agent, steps, layers, replicas, weight_tag, rule_variant, obs_noise_sigma = interactive_menu()
        args.scenario = scenario
        args.mode = mode
        args.agent = agent
        args.steps = steps
        args.layers = layers
        args.weights = weight_tag
        args.variant = rule_variant
        args.obs_noise_sigma = obs_noise_sigma
        if replicas is not None:
            args.replicas = replicas

    # Scenario validation
    if args.mode == "gene_sweep":
        if args.scenario is None:
            args.scenario = "all"
        if args.scenario != "all" and args.scenario not in VALID_SCENARIOS:
            print(f"Scenario '{args.scenario}' invalid. Using 'all'.")
            args.scenario = "all"
    else:
        if args.scenario not in VALID_SCENARIOS:
            print(f"Scenario '{args.scenario}' invalid. Using '{VALID_SCENARIOS[0]}'.")
            args.scenario = VALID_SCENARIOS[0]

    # Weight tag validation
    valid_weight_tags = {t for t, _ in TRAINING_DATASETS}
    args.weights = args.weights.strip().lower()
    if args.weights not in valid_weight_tags:
        print(f"weights '{args.weights}' invalid. Using 'sup_rb_evo_top5_auto'.")
        args.weights = "sup_rb_evo_top5_auto"

    # Agent validation
    valid_agents = {"rule", "random"} | set(MLP_VARIANT_TAGS) | set(LLM_VARIANT_TAGS)
    if args.mode != "gene_sweep" and args.agent not in valid_agents:
        print(f"agent '{args.agent}' invalid. Starting interactive menu.")
        mode, scenario, agent, steps, layers, replicas, weight_tag, rule_variant, obs_noise_sigma = interactive_menu()
        args.scenario = scenario
        args.mode = mode
        args.agent = agent
        args.steps = steps
        args.layers = layers
        args.weights = weight_tag
        args.variant = rule_variant
        args.obs_noise_sigma = obs_noise_sigma
        if replicas is not None:
            args.replicas = replicas

    # Rule variant validation
    if args.variant not in RULE_VARIANT_TAGS:
        print(f"variant '{args.variant}' invalid. Using 'rb_fix_fix'.")
        args.variant = "rb_fix_fix"

    if args.mode == "gui":
        run_gui(
            args.scenario,
            args.agent,
            args.steps,
            args.layers,
            args.weights,
            args.variant,
            max(0.0, args.obs_noise_sigma),
        )
    elif args.mode == "batch":
        run_batch(
            args.scenario,
            args.agent,
            args.steps,
            args.replicas,
            args.layers,
            args.weights,
            args.variant,
        )
    elif args.mode == "gene_sweep":
        selected = VALID_SCENARIOS if args.scenario == "all" else [args.scenario]
        run_rb_gene_sweep_batch(
            steps=args.steps,
            replicas=args.replicas,
            scenarios=selected,
        )
    else:
        print("Invalid mode. Use 'gui', 'batch', or 'gene_sweep'.")
        sys.exit(1)
