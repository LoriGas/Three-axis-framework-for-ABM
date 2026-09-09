# Three-axis framework for agent-based models

This repository contains the simulation code, benchmark pipelines, analysis notebooks, and experimental results associated with a three-axis framework for comparing decision-making strategies in agent-based models (ABMs).

The framework separates agent design along three dimensions:

1. **Policy family** — random, rule-based, multilayer perceptron (MLP), or large language model (LLM).
2. **Policy origin** — fixed parameters, random initialization, offline behavioural cloning, or an externally supplied LLM policy.
3. **Adaptation mechanism** — fixed inheritance, random offspring, evolutionary mutation, or online learning.

Agents inhabit a spatial resource environment in which they move, harvest food, consume energy, reproduce, and die. The experiments compare survival and behavioural outcomes across five environmental regimes.

## Environmental scenarios

| Scenario | Base metabolism | Food regeneration | Description |
| --- | ---: | ---: | --- |
| S1 | 1.949 | 0.042 | Very high metabolism / very low regeneration |
| S2 | 1.205 | 0.173 | High metabolism / medium-low regeneration |
| S3 | 1.119 | 0.191 | Medium metabolism / medium regeneration |
| S4 | 1.036 | 0.202 | Medium-low metabolism / medium-high regeneration |
| S5 | 0.626 | 0.274 | Low metabolism / high regeneration |

The default environment is a 15 × 15 grid initialized with 50 agents. Complete parameter definitions are available in `config.py`.

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
analyze_llm_decisions.ipynb     LLM decision analysis
paper_benchmark_analysis.ipynb  Cross-family benchmark analysis
explore_scenarios_plot.ipynb    Scenario-space visualization
results/                        Paper datasets, logs, summaries, and figures
weights/                        Training-loss histories
```

## Installation

Python 3.10 or newer is recommended.

```bash
git clone https://github.com/LoriGas/Three-axis-framework-for-ABM.git
cd Three-axis-framework-for-ABM

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy pandas torch scikit-learn rich pygame matplotlib seaborn jupyter ipywidgets
```

The repository uses [Git LFS](https://git-lfs.com/) for large JSONL experiment logs. Install Git LFS before cloning, or run the following afterward:

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

Run the main benchmark across scenarios, policy variants, and MLP depths:

```bash
python run_analysis.py
```

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

Then run:

```bash
python run_analysis_llm.py
```

Provider-specific launchers are included for GPT-5 mini, GPT-5 nano, Ollama Cloud, and Alibaba Cloud Model Studio/Qwen. The local `.env` file is ignored by Git and must never be committed.

## Results and notebooks

The `results/` directory contains the data used by the analysis notebooks, including:

- aggregate benchmark summaries;
- per-step time series;
- per-episode simulation outcomes;
- individual LLM decisions and request logs;
- scenario exploration and sensitivity-analysis outputs.

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

If you use this code or the accompanying experimental results, please cite the associated paper. Full bibliographic information will be added when the paper is available.
