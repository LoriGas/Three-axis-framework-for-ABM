"""Run the interactive LLM benchmark against Ollama Cloud.

This is the Ollama counterpart of ``run_analysis_llm.py``.  It deliberately
reuses that benchmark so scenarios, prompts, metrics, CSV schemas, resume
behaviour, and multiprocessing stay identical.  Only the model endpoint and
authentication defaults change.

Run with the default model::

    python run_analysis_ollama.py

Or select another cloud model::

    OLLAMA_MODEL=gpt-oss:120b python run_analysis_ollama.py

The API key is read from ``OLLAMA_API_KEY`` or the project's local ``.env``.
Any explicit benchmark tuning variables such as ``LLM_ANALYSIS_MAX_WORKERS``,
``LLM_TEMPERATURE`` and ``LLM_CACHE_ENABLED`` are passed through unchanged.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
BENCHMARK_SCRIPT = PROJECT_ROOT / "run_analysis_llm.py"
DEFAULT_MODEL = "gemma4:31b"
OLLAMA_CLOUD_URL = "https://ollama.com/v1"


def _local_env_value(key: str) -> str:
    """Read one value from .env without copying unrelated provider secrets."""
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return ""
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        candidate, value = line.split("=", 1)
        if candidate.strip() == key:
            return value.strip().strip("\"'")
    return ""


def _ollama_environment() -> tuple[dict[str, str], str, str]:
    """Build the Ollama Cloud environment without leaking OpenAI settings."""
    env = os.environ.copy()
    model = env.get("OLLAMA_MODEL", "").strip() or DEFAULT_MODEL
    base_url = OLLAMA_CLOUD_URL

    # Override, rather than default, all provider-identifying variables.  This
    # prevents values loaded from the project's .env from routing an Ollama run
    # to an external provider by accident.
    env["LLM_ENABLED"] = "1"
    env["LLM_MODEL"] = model
    env["LLM_BASE_URL"] = base_url
    api_key = (
        env.get("OLLAMA_API_KEY", "").strip()
        or _local_env_value("OLLAMA_API_KEY")
    )
    if not api_key:
        raise ValueError("OLLAMA_API_KEY is missing from the environment and .env")
    env["LLM_API_KEY"] = api_key
    env["LLM_JSON_MODE"] = "1"
    env.setdefault("LLM_MAX_TOKENS", "64")
    env.setdefault("LLM_REASONING_EFFORT", "none")
    env.setdefault("LLM_DECISION_MAX_WORKERS", "1")
    env.setdefault("LLM_RETRY_DELAY", "5.0")
    env.pop("LLM_PROMPT_VARIANT_LOCKED", None)
    return env, model, base_url


def main() -> None:
    try:
        env, model, base_url = _ollama_environment()
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    print(f"\n=== Interactive Ollama Cloud benchmark: {model} ===", flush=True)
    print(f"Endpoint: {base_url}", flush=True)
    print(
        "Using the API key from OLLAMA_API_KEY or the project .env.",
        flush=True,
    )
    try:
        subprocess.run(
            [sys.executable, str(BENCHMARK_SCRIPT)],
            cwd=PROJECT_ROOT,
            env=env,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode) from exc


if __name__ == "__main__":
    main()
