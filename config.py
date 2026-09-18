"""Central simulation configuration."""

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, FrozenSet, List, Tuple

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR: Path = Path(__file__).resolve().parent
WEIGHTS_DIR: Path = BASE_DIR / "weights"
RESULTS_DIR: Path = BASE_DIR / "results"

WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def model_filename_slug(model_name: str) -> str:
    """Return a stable, filesystem-safe identifier for an LLM model name."""
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", model_name.strip()).strip("-._")
    return slug or "unknown-model"

# ---------------------------------------------------------------------------
# Action constants  (paper: N/S/E/W + stay)
# ---------------------------------------------------------------------------
NEIGHBOR_OFFSETS: List[Tuple[int, int]] = [(0, -1), (0, 1), (1, 0), (-1, 0)]
ACTIONS: List[Tuple[int, int]] = NEIGHBOR_OFFSETS + [(0, 0)]

# ---------------------------------------------------------------------------
# Agent parameters
# ---------------------------------------------------------------------------
ALPHA_MIN: float = 0.0   # minimum risk-propensity gene alpha
ALPHA_MAX: float = 2.0   # maximum risk-propensity gene alpha
BETA_MIN: float = 0.0    # minimum reproduction gene beta
BETA_MAX: float = 1.0    # maximum reproduction gene beta


@dataclass
class ModelParams:
    """All configurable simulation parameters."""

    # Grid  (paper: l x l = 15 x 15)
    grid_width: int = 15
    grid_height: int = 15

    # Population  (paper: n = 50, e_i ~ U[1, 5])
    initial_agents: int = 50
    initial_alpha: float = -1.0   # < 0 → sampled uniformly per agent
    initial_beta: float = -1.0    # < 0 -> sampled uniformly per agent

    # Food  (paper: f_max = 8, fc_max = 3)
    food_regen: float = 0.1
    max_food_cell: float = 8.0
    max_bite: float = 3.0

    # Perception uncertainty. Neighbour-food observations receive independent
    # Gaussian noise N(0, sigma^2); 0.0 preserves the original exact perception.
    obs_noise_sigma: float = 0.0

    # LLM system-prompt treatment used by classical (non-uncertainty) runs.
    llm_prompt_variant: str = "goal"

    # Energy / metabolism  (paper: bm scenario-dependent, ft = 0.20)
    base_metabolism: float = 0.1
    transfer_fraction: float = 0.20

    # Reproduction
    reproduction_scale: float = 5.0  # GUI energy reference; not part of the reward
    alpha_sigma: float = 0.05          # sigma for evolutionary alpha mutation
    beta_sigma: float = 0.05           # σ for evolutionary beta mutation

    # Neural network
    net_hidden_layers: int = 1
    net_mutation_rate: float = 0.05    # pm
    net_mutation_strength: float = 0.05  # σ for MLP weight mutation

    # Online learning (shared by RB and MLP agents)
    rl_lr_alpha: float = 0.05          # η_α  (RB adaptive-threshold update)
    rl_lr_beta: float = 0.05           # η_β  (RB beta REINFORCE update)
    rl_lr_net: float = 0.0003          # Adam lr for MLP online learning
    rl_entropy_beta: float = 0.01      # β  entropy regularisation (MLP actor)
    rl_baseline_alpha: float = 0.15    # α  EMA baseline (RuleBasedAgent only)
    rl_reward_clip: float = 2.0        # safety bound; inactive for current shaped rewards
    rl_birth_bonus: float = 0.2        # bb
    rl_survival_bonus: float = 0.02    # b_surv
    rl_energy_scale: float = 3.0       # scale for net-energy tanh reward
    # n-step A2C (MLP online learning only)
    rl_n_steps: int = 4                # trajectory buffer length
    rl_gamma: float = 0.95             # discount factor
    rl_critic_coef: float = 0.5        # λ_c  weight of critic loss vs actor loss


# ---------------------------------------------------------------------------
# Experimental scenarios  (paper: S1–S5, Table 5)
# ---------------------------------------------------------------------------
SCENARIOS: Dict[str, Dict[str, float]] = {
    # Empirical points taken directly from results/explore_scenarios.csv
    # (rb_rand_evo, 500 simulations per point), all with food_regen < 0.3.
    # Survival probabilities in that CSV:
    # S1=0.0%, S2=26.2%, S3=51.0%, S4=72.4%, S5=100.0%.
    # Approximate line: base_metabolism = -5.693 * food_regen + 2.191.
    "S1": {"base_metabolism": 1.9490, "food_regen": 0.0420},
    "S2": {"base_metabolism": 1.2050, "food_regen": 0.1730},
    "S3": {"base_metabolism": 1.1190, "food_regen": 0.1910},
    "S4": {"base_metabolism": 1.0360, "food_regen": 0.2020},
    "S5": {"base_metabolism": 0.6260, "food_regen": 0.2740},
}

VALID_SCENARIOS: List[str] = list(SCENARIOS.keys())


def get_params_for_scenario(name: str) -> ModelParams:
    params = ModelParams()
    for key, value in SCENARIOS.get(name, {}).items():
        setattr(params, key, value)
    return params


# ---------------------------------------------------------------------------
# Training constants
# ---------------------------------------------------------------------------
PERCEPTION_SIZE: int = 10
NUM_ACTIONS: int = 5
DATASET_FLUSH_CHUNK: int = 10_000

# ---------------------------------------------------------------------------
# Agent-mode registry  (paper: variant table, V8--V14)
# ---------------------------------------------------------------------------

# Variants initialised via offline behavioural cloning (V12--V14)
MLP_OFFLINE_INIT: FrozenSet[str] = frozenset({
    "mlp_offline_fix",
    "mlp_offline_evo",
    "mlp_offline_learn",
})

# Variants that apply evolutionary weight mutation across generations
MLP_EVOLUTION: FrozenSet[str] = frozenset({
    "mlp_rand_evo",
    "mlp_offline_evo",
})

# Variants where offspring receive fresh random weights
MLP_RANDOM_OFFSPRING: FrozenSet[str] = frozenset({
    "mlp_rand_rand",
})

# Variants that apply online REINFORCE updates within each agent's lifetime
MLP_ONLINE_LEARNING: FrozenSet[str] = frozenset({
    "mlp_rand_learn",
    "mlp_offline_learn",
})

# ---------------------------------------------------------------------------
# Variant tag lists  (used by GUI and CLI validation)
# ---------------------------------------------------------------------------
RULE_VARIANT_TAGS: List[str] = [
    "rb_rand_fix",     # V1: random founders, exact inheritance
    "rb_rand_rand",    # V2
    "rb_rand_evo",     # V3
    "rb_rand_learn",   # V4
    "rb_fix_fix",      # V5
    "rb_fix_evo",      # V6
    "rb_fix_learn",    # V7
]

MLP_VARIANT_TAGS: List[str] = [
    "mlp_rand_fix",      # V8
    "mlp_rand_rand",     # V9
    "mlp_rand_evo",      # V10
    "mlp_rand_learn",    # V11
    "mlp_offline_fix",   # V12
    "mlp_offline_evo",   # V13
    "mlp_offline_learn", # V14
]

LLM_VARIANT_TAGS: List[str] = [
    "llm",
]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
