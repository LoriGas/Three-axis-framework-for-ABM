"""Dataset generation (supervised rollouts) and K-Fold supervised training."""

import csv
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, TensorDataset

from config import (
    ACTIONS,
    ALPHA_MAX,
    DATASET_FLUSH_CHUNK,
    RESULTS_DIR,
    WEIGHTS_DIR,
    get_params_for_scenario,
)
from agents.random_agent import RandomAgent
from model import World
from network import NeuralNetwork

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ACTION_TO_IDX: Dict[Tuple[int, int], int] = {a: i for i, a in enumerate(ACTIONS)}

# Canonical dataset kinds (paper Table 4 naming)
DATASET_KINDS: List[str] = [
    "rb_rand_fix",         # V1: random founders, exact inheritance
    "rb_rand_rand",        # V2: random alpha, random offspring
    "rb_rand_evo",         # V3: random alpha, evolutionary
    "rb_fix_fix",          # V5: fixed alpha=1, no offspring mutation
    "rb_fix_evo",          # V6: fixed alpha=1, evolutionary
    "rb_rand_evo_top5",    # V3-top5: evolutionary, only top 5% agents by survival
]
DATASET_KIND_SET = set(DATASET_KINDS)
FEATURE_COLUMNS: List[str] = [f"in_{i}" for i in range(10)]
TARGET_COLUMNS: List[str] = ["move_target", "repro_target"]
METADATA_COLUMNS: List[str] = ["episode_id", "agent_id"]
DATASET_HEADER: List[str] = FEATURE_COLUMNS + TARGET_COLUMNS + METADATA_COLUMNS

DEFAULT_TRAIN_BATCH_SIZE: int = 256
DEFAULT_VAL_BATCH_SIZE: int = 512
DEFAULT_CV_FOLDS: int = 5

_OPTIMAL_ALPHA_CACHE: Dict[Tuple, float] = {}


def normalize_dataset_kind(kind: str) -> str:
    return kind.lower().strip()


def _teacher_rule_action(
    percept: List[float],
    energy: float,
    transfer_fraction: float,
    alpha: float,
    beta: float,
) -> Tuple[Tuple[int, int], int]:
    """Fast teacher action equivalent to RuleBasedAgent.decide_action for fixed alpha/beta."""
    food_neighbors = percept[:4]
    occ_neighbors = percept[4:8]
    food_here = percept[8]

    best_idx: Optional[int] = None
    best_food = -1.0
    for i in range(4):
        if occ_neighbors[i] > 0.5:
            continue
        f = food_neighbors[i]
        if f > best_food or (f == best_food and random.random() < 0.5):
            best_food = f
            best_idx = i

    reproduce = 1 if random.random() < beta else 0
    if best_idx is None:
        return ACTIONS[4], reproduce

    threshold = food_here + energy * transfer_fraction
    if best_food * alpha > threshold:
        return ACTIONS[best_idx], reproduce
    return ACTIONS[4], reproduce


# ==========================================================================
# Rollout collection
# ==========================================================================

def _rollout_and_log(
    scenario: str,
    policy_type: str,
    out_csv: Path,
    fixed_alpha: Optional[float] = None,
    fixed_beta: Optional[float] = None,
    mutate_propensity: bool = True,
    random_offspring_propensity: bool = False,
    min_rows: int = 500_000,
    max_steps: int = 2000,
    seed0: int = 123,
    min_survival: int = 0,
    obs_noise_sigma: float = 0.0,
) -> None:
    """Run episodes until *min_rows* state-action rows are collected."""
    params = get_params_for_scenario(scenario)
    params.obs_noise_sigma = max(0.0, float(obs_noise_sigma))
    if fixed_alpha is not None:
        params.initial_alpha = float(fixed_alpha)
    if fixed_beta is not None:
        params.initial_beta = float(fixed_beta)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    CHUNK = DATASET_FLUSH_CHUNK

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(DATASET_HEADER)
        buffer: List[List[float]] = []
        rows_written = 0
        ep = 0
        empty_episodes = 0

        while rows_written < min_rows:
            episode_seed = seed0 + ep
            random.seed(episode_seed)
            np.random.seed(episode_seed)
            torch.manual_seed(episode_seed)

            world = World(
                params,
                agent_type="rule",
                mutate_propensity=mutate_propensity,
                random_offspring_propensity=random_offspring_propensity,
            )

            if policy_type == "random":
                world.agents = [RandomAgent(a.x, a.y, a.energy) for a in world.agents]
            elif policy_type not in {"rb", "rb_fixed"}:
                raise ValueError(f"Unknown policy_type: {policy_type}")

            episode_rows = 0
            agent_ids: dict = {}
            next_agent_id = 0

            for _ in range(max_steps):
                if not world.agents:
                    break

                world.step()

                for agent, percept, action in world.decision_records_step:
                    if agent.age < min_survival:
                        continue

                    if agent not in agent_ids:
                        agent_ids[agent] = next_agent_id
                        next_agent_id += 1

                    move = (action[0], action[1])
                    idx = ACTION_TO_IDX.get(move)
                    if idx is None:
                        continue

                    reproduce = action[2] if len(action) > 2 else False
                    buffer.append(
                        percept
                        + [idx, int(bool(reproduce)), ep, agent_ids[agent]]
                    )
                    episode_rows += 1

                if len(buffer) >= CHUNK:
                    writer.writerows(buffer)
                    rows_written += len(buffer)
                    buffer.clear()

            if episode_rows:
                empty_episodes = 0
            else:
                empty_episodes += 1
                if empty_episodes >= 1000:
                    raise RuntimeError(
                        "No useful rows found in 1000 consecutive episodes."
                    )

            ep += 1

        if buffer:
            writer.writerows(buffer)
            rows_written += len(buffer)

    logger.info("[DATASET] %s: %s rows (%d episodes)", out_csv.name, f"{rows_written:,}", ep)


def _rollout_and_log_evo_top5(
    scenario: str,
    out_csv: Path,
    top_pct: float = 0.05,
    min_rows: int = 500_000,
    min_episodes: int = 10,
    max_rows_per_episode: int = 12_000,
    max_steps: int = 2000,
    seed0: int = 789,
    min_survival: int = 0,
    obs_noise_sigma: float = 0.0,
) -> None:
    """Log executed decisions from long-lived evolutionary rule agents.

    Selection is performed independently within each episode.  Per-episode
    sampling prevents one long, high-population run from dominating the dataset
    and guarantees enough episode groups for leakage-free validation.
    """
    params = get_params_for_scenario(scenario)
    params.obs_noise_sigma = max(0.0, float(obs_noise_sigma))
    params.initial_alpha = -1.0  # uniform random alpha per agent

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(DATASET_HEADER)
        rows_written = 0
        ep = 0
        empty_episodes = 0

        while rows_written < min_rows or ep < min_episodes:
            episode_seed = seed0 + ep
            random.seed(episode_seed)
            np.random.seed(episode_seed)
            torch.manual_seed(episode_seed)

            world = World(params, agent_type="rule", mutate_propensity=True)

            # agent_obj -> list[row]  — hard refs prevent GC of dead agents
            agent_rows: dict = {}
            agent_max_age: dict = {}  # agent_obj -> int
            agent_ids: dict = {}
            next_agent_id = 0

            for _ in range(max_steps):
                if not world.agents:
                    break

                world.step()

                for agent, percept, action in world.decision_records_step:
                    if agent.age < min_survival:
                        continue

                    if agent not in agent_ids:
                        agent_ids[agent] = next_agent_id
                        next_agent_id += 1

                    move = (action[0], action[1])
                    idx = ACTION_TO_IDX.get(move)
                    if idx is None:
                        continue

                    reproduce = action[2] if len(action) > 2 else False
                    row = percept + [
                        idx,
                        int(bool(reproduce)),
                        ep,
                        agent_ids[agent],
                    ]

                    if agent not in agent_rows:
                        agent_rows[agent] = []
                        agent_max_age[agent] = 0
                    agent_rows[agent].append(row)
                    if agent.age > agent_max_age[agent]:
                        agent_max_age[agent] = agent.age

            if not agent_max_age:
                empty_episodes += 1
                if empty_episodes >= 1000:
                    raise RuntimeError("No useful rows in 1000 consecutive episodes.")
                ep += 1
                continue

            empty_episodes = 0

            # Keep only the top top_pct fraction by max age
            ages = np.array(list(agent_max_age.values()), dtype=np.float64)
            threshold = float(np.percentile(ages, 100.0 * (1.0 - top_pct)))

            selected: List[List[float]] = []
            for agent, rows in agent_rows.items():
                if agent_max_age[agent] >= threshold:
                    selected.extend(rows)

            if max_rows_per_episode > 0 and len(selected) > max_rows_per_episode:
                rng = np.random.default_rng(1_000_000 + episode_seed)
                keep = rng.choice(
                    len(selected), size=max_rows_per_episode, replace=False
                )
                selected = [selected[int(i)] for i in np.sort(keep)]

            if selected:
                writer.writerows(selected)
                rows_written += len(selected)

            ep += 1

    logger.info("[DATASET] %s: %s rows (%d episodes)", out_csv.name, f"{rows_written:,}", ep)


# ==========================================================================
# Optimal alpha estimation
# ==========================================================================

def _rb_evo_alpha_episode(
    args: Tuple[str, int, int]
) -> Tuple[float, int, float]:
    scenario, seed, max_steps = args
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    params = get_params_for_scenario(scenario)
    params.initial_alpha = -1.0
    world = World(params, agent_type="rule", mutate_propensity=True)

    last_mean = float("nan")
    for _ in range(max_steps):
        if world.agents:
            alphas = [
                a.alpha
                for a in world.agents
                if getattr(a, "alive", False) and hasattr(a, "alpha")
            ]
            if alphas:
                last_mean = float(np.mean(alphas))
        world.step()
        if not world.agents:
            break

    final_alphas = [a.alpha for a in world.agents if hasattr(a, "alpha")]
    if final_alphas:
        return float(sum(final_alphas)), len(final_alphas), float(np.mean(final_alphas))
    return 0.0, 0, last_mean


def _mean_final_alpha_rb_evo(
    scenario: str,
    n_episodes: int = 500,
    max_steps: int = 2000,
    max_workers: int = 8,
) -> float:
    import multiprocessing
    import torch.multiprocessing as mp

    tasks = [(scenario, 50_000 + ep, max_steps) for ep in range(n_episodes)]

    if max_workers > 1 and not multiprocessing.current_process().daemon:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=max_workers) as pool:
            results = pool.map(_rb_evo_alpha_episode, tasks)
    else:
        results = [_rb_evo_alpha_episode(t) for t in tasks]

    total_sum = sum(s for s, n, _ in results if n > 0)
    total_n = sum(n for _, n, _ in results if n > 0)
    if total_n > 0:
        return float(total_sum / total_n)

    last_means = [m for _, _, m in results if not np.isnan(m)]
    if last_means:
        return float(np.mean(last_means))

    raise RuntimeError(f"Cannot estimate final alpha for scenario '{scenario}'.")


def estimate_optimal_alpha(
    scenario: str,
    n_episodes: int = 500,
    max_steps: int = 2000,
    max_workers: int = 8,
) -> float:
    cache_key = (scenario, n_episodes, max_steps, max_workers)
    if cache_key in _OPTIMAL_ALPHA_CACHE:
        return _OPTIMAL_ALPHA_CACHE[cache_key]
    result = _mean_final_alpha_rb_evo(
        scenario, n_episodes=n_episodes, max_steps=max_steps, max_workers=max_workers
    )
    _OPTIMAL_ALPHA_CACHE[cache_key] = float(result)
    return float(result)


# ==========================================================================
# Dataset generation
# ==========================================================================

def generate_dataset(
    scenario: str,
    kind: str,
    best_alpha: Optional[float] = None,
    min_rows: int = 500_000,
    min_episodes: int = 10,
    max_rows_per_episode: int = 12_000,
    max_steps: int = 2000,
    min_survival: int = 50,
    obs_noise_sigma: float = 0.0,
    output_kind: Optional[str] = None,
) -> Path:
    """Generate a supervised dataset for the given scenario and kind."""
    kind = normalize_dataset_kind(kind)
    if kind not in DATASET_KIND_SET:
        raise ValueError(f"kind must be one of: {', '.join(DATASET_KINDS)}")

    if best_alpha is None and kind in {"rb_fix_fix", "rb_fix_evo"}:
        best_alpha = estimate_optimal_alpha(scenario)

    output_kind = normalize_dataset_kind(output_kind) if output_kind else kind
    out_csv = RESULTS_DIR / f"dataset_{scenario}_{output_kind}.csv"

    if kind == "rb_rand_fix":
        _rollout_and_log(scenario, "rb", out_csv,
                         fixed_alpha=None, mutate_propensity=False,
                         min_rows=min_rows, max_steps=max_steps, min_survival=min_survival,
                         obs_noise_sigma=obs_noise_sigma)
    elif kind == "rb_rand_rand":
        _rollout_and_log(scenario, "rb", out_csv,
                         fixed_alpha=None, mutate_propensity=False,
                         random_offspring_propensity=True,
                         min_rows=min_rows, max_steps=max_steps, min_survival=min_survival,
                         obs_noise_sigma=obs_noise_sigma)
    elif kind == "rb_rand_evo":
        _rollout_and_log(scenario, "rb", out_csv,
                         fixed_alpha=None, mutate_propensity=True,
                         min_rows=min_rows, max_steps=max_steps, min_survival=min_survival,
                         obs_noise_sigma=obs_noise_sigma)
    elif kind == "rb_fix_fix":
        _rollout_and_log(scenario, "rb_fixed", out_csv,
                         fixed_alpha=best_alpha, fixed_beta=0.5, mutate_propensity=False,
                         min_rows=min_rows, max_steps=max_steps, min_survival=min_survival,
                         obs_noise_sigma=obs_noise_sigma)
    elif kind == "rb_rand_evo_top5":
        _rollout_and_log_evo_top5(scenario, out_csv,
                                  top_pct=0.05,
                                  min_rows=min_rows,
                                  min_episodes=min_episodes,
                                  max_rows_per_episode=max_rows_per_episode,
                                  max_steps=max_steps,
                                  min_survival=min_survival,
                                  obs_noise_sigma=obs_noise_sigma)
    else:  # rb_fix_evo
        _rollout_and_log(scenario, "rb_fixed", out_csv,
                         fixed_alpha=best_alpha, fixed_beta=0.5, mutate_propensity=True,
                         min_rows=min_rows, max_steps=max_steps, min_survival=min_survival,
                         obs_noise_sigma=obs_noise_sigma)

    return out_csv


def _dataset_path(scenario: str, kind: str) -> Path:
    return RESULTS_DIR / f"dataset_{scenario}_{normalize_dataset_kind(kind)}.csv"


def merge_datasets_for_scenario(
    scenario: str,
    input_kinds: List[str],
    output_kind: str,
) -> Path:
    """Concatenate multiple per-scenario datasets into a single output dataset."""
    if not input_kinds:
        raise ValueError("input_kinds must contain at least one dataset kind")

    out = _dataset_path(scenario, output_kind)
    out.parent.mkdir(parents=True, exist_ok=True)

    header: Optional[List[str]] = None
    rows_written = 0

    with open(out, "w", newline="") as fout:
        writer = csv.writer(fout)

        for kind in input_kinds:
            src = _dataset_path(scenario, kind)
            if not src.exists():
                raise FileNotFoundError(f"Dataset not found: {src}")

            with open(src, "r", newline="") as fin:
                reader = csv.reader(fin)
                try:
                    src_header = next(reader)
                except StopIteration:
                    continue

                if header is None:
                    header = src_header
                    writer.writerow(header)
                elif src_header != header:
                    raise ValueError(
                        f"Incompatible dataset header in {src.name}: expected {header}, got {src_header}"
                    )

                for row in reader:
                    writer.writerow(row)
                    rows_written += 1

    if header is None:
        raise ValueError("No rows available to merge: all input datasets are empty.")

    logger.info(
        "[DAGGER] merged %d datasets -> %s rows (%s)",
        len(input_kinds),
        f"{rows_written:,}",
        out.name,
    )
    return out


def collect_dagger_dataset(
    scenario: str,
    n_layers: int,
    source_weight_tag: str,
    output_kind: str,
    min_rows: int = 50_000,
    max_steps: int = 2000,
    min_survival: int = 0,
    seed0: int = 33_000,
    only_disagreements: bool = True,
    teacher_alpha: Optional[float] = None,
    teacher_beta: float = 0.5,
) -> Path:
    """Collect learner-visited states and label them with a rule-based teacher (DAgger)."""
    source_weight_tag = source_weight_tag.strip().lower()
    out_csv = _dataset_path(scenario, output_kind)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    weights_path = WEIGHTS_DIR / f"weights_{scenario}_L{n_layers}_{source_weight_tag}.pth"
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found for DAgger collection: {weights_path}")

    params = get_params_for_scenario(scenario)
    params.net_hidden_layers = int(n_layers)

    learner_net = NeuralNetwork(num_hidden_layers=n_layers)
    learner_net.load_state_dict(torch.load(weights_path, weights_only=True))
    learner_net.eval()

    if teacher_alpha is None:
        teacher_alpha = estimate_optimal_alpha(scenario)
    teacher_alpha = max(0.0, min(ALPHA_MAX, float(teacher_alpha)))
    teacher_beta = max(0.0, min(1.0, float(teacher_beta)))
    transfer_fraction = float(params.transfer_fraction)

    CHUNK = DATASET_FLUSH_CHUNK
    rows_written = 0
    ep = 0
    empty_episodes = 0

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(DATASET_HEADER)
        buffer: List[List[float]] = []

        while rows_written < min_rows:
            episode_seed = seed0 + ep
            random.seed(episode_seed)
            np.random.seed(episode_seed)
            torch.manual_seed(episode_seed)

            world = World(
                params,
                agent_type="mlp_offline_fix",
                net_template=learner_net,
            )

            episode_rows = 0
            agent_ids: dict = {}
            next_agent_id = 0
            for _ in range(max_steps):
                if not world.agents:
                    break

                # Advance the learner once, then label the exact percepts used
                # in its real action phase.  This keeps DAgger consistent with
                # the supervised rollout collector.
                world.step()

                for agent, percept, learner_action in world.decision_records_step:
                    if agent.age < min_survival:
                        continue

                    if agent not in agent_ids:
                        agent_ids[agent] = next_agent_id
                        next_agent_id += 1

                    l_move = (learner_action[0], learner_action[1])
                    l_repro = int(bool(learner_action[2] if len(learner_action) > 2 else False))

                    t_move, t_repro = _teacher_rule_action(
                        percept=percept,
                        energy=float(percept[9]),
                        transfer_fraction=transfer_fraction,
                        alpha=teacher_alpha,
                        beta=teacher_beta,
                    )

                    if only_disagreements and l_move == t_move and l_repro == t_repro:
                        continue

                    t_idx = ACTION_TO_IDX.get(t_move)
                    if t_idx is None:
                        continue
                    buffer.append(
                        percept + [t_idx, t_repro, ep, agent_ids[agent]]
                    )
                    episode_rows += 1

                if len(buffer) >= CHUNK:
                    writer.writerows(buffer)
                    rows_written += len(buffer)
                    buffer.clear()

            if episode_rows == 0:
                empty_episodes += 1
                if empty_episodes >= 1000:
                    raise RuntimeError("No DAgger rows collected in 1000 consecutive episodes.")
            else:
                empty_episodes = 0

            ep += 1

        if buffer:
            writer.writerows(buffer)
            rows_written += len(buffer)

    logger.info(
        "[DAGGER] %s: %s rows (%d episodes, only_disagreements=%s)",
        out_csv.name,
        f"{rows_written:,}",
        ep,
        only_disagreements,
    )
    return out_csv


# ==========================================================================
# Supervised training with K-Fold CV
# ==========================================================================

def train(
    scenario: str,
    n_layers: int,
    dataset_kind: str,
    epochs: int = 80,
    patience: int = 10,
    seed: int = 42,
    output_weight_tag: Optional[str] = None,
    train_batch_size: int = DEFAULT_TRAIN_BATCH_SIZE,
    val_batch_size: int = DEFAULT_VAL_BATCH_SIZE,
    cv_folds: int = DEFAULT_CV_FOLDS,
) -> Tuple[Path, float, float]:
    """Episode-grouped stratified CV followed by a full-data refit.

    Returns the refitted weight path and mean held-out movement and
    reproduction accuracies across folds.
    """
    dataset_kind = normalize_dataset_kind(dataset_kind)
    weight_tag = output_weight_tag.strip().lower() if output_weight_tag else f"sup_{dataset_kind}"
    f_out = WEIGHTS_DIR / f"weights_{scenario}_L{n_layers}_{weight_tag}.pth"
    csv_path = RESULTS_DIR / f"dataset_{scenario}_{dataset_kind}.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"Empty dataset: {csv_path}")

    required = FEATURE_COLUMNS + TARGET_COLUMNS + METADATA_COLUMNS
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(
            "Dataset lacks grouped-validation metadata; regenerate it first: "
            + ", ".join(missing)
        )

    X_np = df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y_move_np = df["move_target"].to_numpy(dtype=np.int64)
    y_repro_np = df["repro_target"].to_numpy(dtype=np.int64)
    groups_np = df["episode_id"].to_numpy(dtype=np.int64)

    class_counts = np.bincount(y_move_np)
    min_class = int(class_counts[class_counts > 0].min())
    n_groups = int(np.unique(groups_np).size)
    effective_folds = min(cv_folds, min_class, n_groups)
    if effective_folds < 2:
        raise ValueError(
            "Grouped K-Fold requires at least two episodes and two samples "
            "in every movement class."
        )

    skf = StratifiedGroupKFold(
        n_splits=effective_folds, shuffle=True, random_state=seed
    )

    cv_rows: list = []
    fold_best_move_acc: list = []
    fold_best_repro_acc: list = []
    fold_best_epochs: list[int] = []

    for fold_idx, (train_idx, val_idx) in enumerate(
        skf.split(X_np, y_move_np, groups=groups_np), start=1
    ):
        X_train = torch.from_numpy(X_np[train_idx]).float()
        y_move_train = torch.from_numpy(y_move_np[train_idx]).long()
        y_repro_train = torch.from_numpy(y_repro_np[train_idx]).float()
        X_val = torch.from_numpy(X_np[val_idx]).float()
        y_move_val = torch.from_numpy(y_move_np[val_idx]).long()
        y_repro_val = torch.from_numpy(y_repro_np[val_idx]).float()

        train_loader = DataLoader(
            TensorDataset(X_train, y_move_train, y_repro_train), batch_size=train_batch_size,
            shuffle=True, generator=torch.Generator().manual_seed(seed + fold_idx),
        )
        val_loader = DataLoader(
            TensorDataset(X_val, y_move_val, y_repro_val), batch_size=val_batch_size, shuffle=False
        )

        logger.info("[CV %d/%d] %s train / %s val",
                    fold_idx, effective_folds, f"{len(X_train):,}", f"{len(X_val):,}")

        torch.manual_seed(seed + fold_idx)
        net = NeuralNetwork(num_hidden_layers=n_layers)
        optimizer = optim.Adam(net.parameters(), lr=0.001)
        criterion_move = nn.CrossEntropyLoss()
        pos_count = float(y_repro_train.sum().item())
        neg_count = float(len(y_repro_train) - pos_count)
        pos_weight = torch.tensor([neg_count / max(pos_count, 1.0)], dtype=torch.float32)
        criterion_repro = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        best_fold_val_loss = float("inf")
        best_fold_state = None
        no_improve = 0
        best_fold_move_acc = float("nan")
        best_fold_repro_acc = float("nan")
        best_fold_epoch = 1

        for epoch in range(epochs):
            net.train()
            train_loss, train_move_loss, train_repro_loss = 0.0, 0.0, 0.0
            train_move_correct, train_move_total = 0, 0
            train_repro_correct, train_repro_total = 0, 0
            n_batches = 0
            for bx, by_move, by_repro in train_loader:
                optimizer.zero_grad()
                logits = net(bx)
                move_logits = logits[:, :5]
                repro_logits = logits[:, 5]
                loss_move = criterion_move(move_logits, by_move)
                loss_repro = criterion_repro(repro_logits, by_repro)
                loss = loss_move + loss_repro
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                train_move_loss += loss_move.item()
                train_repro_loss += loss_repro.item()
                n_batches += 1
                train_move_correct += (move_logits.argmax(dim=1) == by_move).sum().item()
                train_move_total += by_move.size(0)
                repro_pred = (torch.sigmoid(repro_logits) >= 0.5).long()
                train_repro_correct += (repro_pred == by_repro.long()).sum().item()
                train_repro_total += by_repro.size(0)

            net.eval()
            val_loss, val_move_loss, val_repro_loss = 0.0, 0.0, 0.0
            val_move_correct, val_move_total = 0, 0
            val_repro_correct, val_repro_total = 0, 0
            val_batches = 0
            with torch.no_grad():
                for bx, by_move, by_repro in val_loader:
                    logits = net(bx)
                    move_logits = logits[:, :5]
                    repro_logits = logits[:, 5]
                    loss_move = criterion_move(move_logits, by_move)
                    loss_repro = criterion_repro(repro_logits, by_repro)
                    val_loss += (loss_move + loss_repro).item()
                    val_move_loss += loss_move.item()
                    val_repro_loss += loss_repro.item()
                    val_batches += 1
                    val_move_correct += (move_logits.argmax(dim=1) == by_move).sum().item()
                    val_move_total += by_move.size(0)
                    repro_pred = (torch.sigmoid(repro_logits) >= 0.5).long()
                    val_repro_correct += (repro_pred == by_repro.long()).sum().item()
                    val_repro_total += by_repro.size(0)

            avg_val_loss = val_loss / val_batches
            avg_val_move_acc = val_move_correct / val_move_total
            avg_val_repro_acc = val_repro_correct / val_repro_total
            cv_rows.append((
                fold_idx, epoch + 1,
                train_loss / n_batches,
                train_move_loss / n_batches,
                train_repro_loss / n_batches,
                train_move_correct / train_move_total,
                train_repro_correct / train_repro_total,
                avg_val_loss,
                val_move_loss / val_batches,
                val_repro_loss / val_batches,
                avg_val_move_acc,
                avg_val_repro_acc,
            ))

            if avg_val_loss < best_fold_val_loss:
                best_fold_val_loss = avg_val_loss
                best_fold_state = {k: v.clone() for k, v in net.state_dict().items()}
                best_fold_move_acc = avg_val_move_acc
                best_fold_repro_acc = avg_val_repro_acc
                best_fold_epoch = epoch + 1
                no_improve = 0
            else:
                no_improve += 1

            if (epoch + 1) % 10 == 0 or epoch == 0:
                logger.info(
                    "[CV %d/%d] Epoch %3d  val_loss=%.4f  move_acc=%.4f  repro_acc=%.4f",
                    fold_idx, effective_folds, epoch + 1, avg_val_loss, avg_val_move_acc, avg_val_repro_acc
                )

            if patience > 0 and no_improve >= patience:
                logger.info("[CV %d/%d] Early stop epoch %d", fold_idx, effective_folds, epoch + 1)
                break

        if best_fold_state is not None:
            net.load_state_dict(best_fold_state)

        fold_best_move_acc.append(best_fold_move_acc)
        fold_best_repro_acc.append(best_fold_repro_acc)
        fold_best_epochs.append(best_fold_epoch)

    if not fold_best_epochs:
        raise RuntimeError("Grouped K-Fold training failed: no valid fold found.")

    # Cross-validation estimates generalisation and selects the training
    # duration.  The deployed checkpoint is then fitted once on all episodes,
    # rather than being copied from a fold that omitted part of the data.
    final_epochs = max(1, int(round(float(np.median(fold_best_epochs)))))
    torch.manual_seed(seed + 10_000)
    final_net = NeuralNetwork(num_hidden_layers=n_layers)
    final_optimizer = optim.Adam(final_net.parameters(), lr=0.001)
    full_x = torch.from_numpy(X_np).float()
    full_move = torch.from_numpy(y_move_np).long()
    full_repro = torch.from_numpy(y_repro_np).float()
    full_pos = float(full_repro.sum().item())
    full_neg = float(len(full_repro) - full_pos)
    final_move_loss = nn.CrossEntropyLoss()
    final_repro_loss = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            [full_neg / max(full_pos, 1.0)], dtype=torch.float32
        )
    )
    full_loader = DataLoader(
        TensorDataset(full_x, full_move, full_repro),
        batch_size=train_batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed + 10_000),
    )
    final_net.train()
    for _ in range(final_epochs):
        for bx, by_move, by_repro in full_loader:
            final_optimizer.zero_grad()
            logits = final_net(bx)
            loss = final_move_loss(logits[:, :5], by_move)
            loss += final_repro_loss(logits[:, 5], by_repro)
            loss.backward()
            final_optimizer.step()
    final_net.eval()
    torch.save(final_net.state_dict(), f_out)
    logger.info(
        "[SUP] Grouped CV over %d episodes; refit %d epochs on all rows -> %s",
        n_groups,
        final_epochs,
        f_out.name,
    )

    import csv as _csv
    loss_file = f_out.with_suffix(".loss.csv")
    with open(loss_file, "w", newline="") as lf:
        w = _csv.writer(lf)
        w.writerow([
            "fold", "epoch",
            "train_loss", "train_move_loss", "train_repro_loss",
            "train_move_acc", "train_repro_acc",
            "val_loss", "val_move_loss", "val_repro_loss",
            "val_move_acc", "val_repro_acc",
        ])
        for row in cv_rows:
            w.writerow([row[0], row[1]] + [f"{v:.6f}" for v in row[2:]])

    finite_move_acc = [v for v in fold_best_move_acc if np.isfinite(v)]
    finite_repro_acc = [v for v in fold_best_repro_acc if np.isfinite(v)]
    mean_move_acc = float(np.mean(finite_move_acc)) if finite_move_acc else float("nan")
    mean_repro_acc = float(np.mean(finite_repro_acc)) if finite_repro_acc else float("nan")
    return f_out, mean_move_acc, mean_repro_acc
