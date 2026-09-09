"""Run the interactive LLM benchmark with GPT-5 mini.

This is the GPT-5 mini entry point for ``run_analysis_llm.py``. The terminal
asks for the prompt treatment (goal/no_goal), append/rewrite mode, and the
number of new simulations for every scenario. Output filenames remain isolated
from GPT-4o mini through the ``gpt-5-mini`` model suffix.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
BENCHMARK_SCRIPT = PROJECT_ROOT / "run_analysis_llm.py"
MODEL = "gpt-5-mini"


def main() -> None:
    env = os.environ.copy()
    env.setdefault("LLM_ENABLED", "1")
    env["LLM_MODEL"] = MODEL
    env.pop("LLM_PROMPT_VARIANT_LOCKED", None)
    print(f"\n=== Interactive benchmark: {MODEL} ===", flush=True)
    subprocess.run(
        [sys.executable, str(BENCHMARK_SCRIPT)],
        cwd=PROJECT_ROOT,
        env=env,
        check=True,
    )


if __name__ == "__main__":
    main()
