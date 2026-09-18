# Three-axis framework for agent-based models

This repository contains the simulation code, benchmark pipelines, analysis notebooks, and experimental results associated with a three-axis framework for comparing decision-making strategies in agent-based models (ABMs).

The framework locates each experimental configuration along three categorical dimensions:

1. **Decision-making architecture** — random, rule-based, trainable model, or large language model (LLM).
2. **Behavioural initialisation** — fixed, random-sampled, or offline learned.
3. **Behavioural change mechanism** — no change, random resampling, evolution, or online learning.

Agents inhabit a spatial resource environment in which they move, harvest food, consume energy, reproduce, and die. The experiments compare survival and behavioural outcomes across five environmental regimes.

## Environmental scenarios

| Scenario | Base metabolism | Food regeneration | Description |
| --- | ---: | ---: | --- |
| S1 | 1.949 | 0.042 | Very high metabolism / very low regeneration |
| S2 | 1.205 | 0.173 | High metabolism / medium-low regeneration |
| S3 | 1.119 | 0.191 | Medium metabolism / medium regeneration |
| S4 | 1.036 | 0.202 | Medium-low metabolism / medium-high regeneration |
| S5 | 0.626 | 0.274 | Low metabolism / high regeneration |

The default environment is a 15 × 15 grid initialised with 50 agents. Complete parameter definitions are available in `config.py`.

## Repository structure

```text
agents/                         Agent implementations
config.py                       Model parameters, scenarios, and variant registry
model.py                        World dynamics and simulation state
network.py                      Neural-network architecture
simulation.py                   Headless batch and parameter-sweep utilities
gui.py                          Interactive graphical simulation
main.py                         Main command-line entry point
lib_training.py                 Dataset generation and supervised training
run_analysis.py                 Classical-agent benchmark pipeline
run_analysis_llm.py             LLM-agent benchmark pipeline
run_automation.py               Dataset-generation and MLP-training pipeline
run_rule_alpha_beta_sensitivity.py
                                Rule-based sensitivity analysis
paper_benchmark_analysis.ipynb  Cross-family benchmark analysis
explore_scenarios_plot.ipynb    Scenario-space visualization
results/                        Paper datasets, logs, summaries, and figures
weights/                        Behavioural-cloning checkpoints and loss histories
```

## Installation

Python 3.11 or newer is recommended. The exact direct-dependency versions used
to validate this repository are recorded in `requirements.txt`.

```bash
git clone https://github.com/LoriGas/Three-axis-framework-for-ABM.git
cd Three-axis-framework-for-ABM

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The repository uses [Git LFS](https://git-lfs.com/) for large JSONL experiment
logs. Install Git LFS before cloning, or run the following afterward. A complete
checkout is several gigabytes because the decision-level paper results are
included.

```bash
git lfs install
git lfs pull
```

## Quick start

Launch the interactive menu:

```bash
python main.py
```

Run a graphical simulation:

```bash
python main.py --mode gui --scenario S3 --agent rule --variant rb_rand_evo --steps 500
```

Run a headless batch:

```bash
python main.py --mode batch --scenario S3 --agent rule --variant rb_rand_evo --steps 500 --replicas 5
```

Run the rule-based gene sweep across all scenarios:

```bash
python main.py --mode gene_sweep --scenario all --steps 500 --replicas 5
```

## Reproducing the experiments

Generate supervised datasets and train the MLP policies:

```bash
python run_automation.py
```

Choose option `3` to regenerate both the behavioural-cloning datasets and all
15 MLP checkpoints. The exact checkpoints used for the committed results are
also provided under `weights/`.

Run the main benchmark across scenarios, policy variants, and MLP depths:

```bash
python run_analysis.py
```

Choose `a` when prompted to run the complete paper benchmark: 300 episodes per
condition, a 2,000-step horizon, and MLP depths 1–3.

Run the rule-based alpha–beta sensitivity analysis:

```bash
python run_rule_alpha_beta_sensitivity.py
```

The scripts use multiprocessing and may be computationally expensive. Their default episode counts, worker counts, and maximum horizons are defined near the top of each script.

## LLM experiments

LLM runs use an OpenAI-compatible chat-completions endpoint. Copy the example environment file and provide credentials locally:

```bash
cp .env.example .env
```

At minimum, configure:

```text
LLM_ENABLED=1
LLM_API_KEY=your_key_here
LLM_MODEL=your_model_name
```

The paper's LLM panel uses five valid replications for every
model–prompt–scenario cell and a 400-step horizon. The benchmark defaults now
match that design. For example:

```bash
LLM_TARGET_EPISODES_PER_SCENARIO=5 python run_analysis_llm.py
```

Set `LLM_MODEL=gpt-5-nano` to reproduce the GPT-5 nano treatment. Dedicated
launchers are included for GPT-5 mini, Ollama Cloud, and Alibaba Cloud Model
Studio/Qwen. The Qwen launcher defaults to the paper model,
`qwen3.7-flash-2026-07-15`. The local `.env` file is ignored by Git and must
never be committed.

## Results and notebooks

The `results/` directory contains the data used by the analysis notebooks, including:

- aggregate benchmark summaries;
- per-step time series;
- per-episode simulation outcomes;
- individual LLM decisions and request logs;
- scenario exploration and sensitivity-analysis outputs.

The canonical analyses read only the files directly under `results/`. The
`results/archive/gpt5_nano_no_goal_20260818_legacy_invalid/` directory is
retained solely as an audit record of an invalid legacy run and is excluded
from every result reported in the paper.

Open the notebooks with:

```bash
jupyter notebook
```

The notebooks locate the repository from the current working directory and read the committed files under `results/`.

## Tests

Run the complete test suite with:

```bash
python -m unittest discover -v
```

The tests cover dataset collection, LLM decision schemas and failure handling, online learning, and rule-based inheritance behaviour.

## Citation

If you use this code or the accompanying experimental results, please cite:

> Gastaldo, L., and Bertolotti, F. (2026). *A Framework for Comparing
> Rule-Based, MLP-Based and LLM-Based Agents in Agent-Based Modelling*.
> Manuscript.

Machine-readable citation metadata are provided in `CITATION.cff`.
