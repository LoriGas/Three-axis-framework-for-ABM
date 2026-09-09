"""Run the interactive LLM benchmark on Alibaba Cloud Model Studio (Qwen).

The simulation, prompts, CSV schemas, append/rewrite workflow, quality control,
and incremental saving are provided by ``run_analysis_llm.py``.  This launcher
only supplies the Qwen endpoint, credentials, and inference defaults.

Required credential (shell or project ``.env``)::

    DASHSCOPE_API_KEY=...

``QWEN_API_KEY`` is accepted as an alternative.  The current Alibaba Cloud
documentation recommends a workspace-specific base URL.  Set the URL shown in
your Model Studio workspace, for example::

    QWEN_BASE_URL=https://WORKSPACE_ID.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1

If no URL is supplied, the launcher uses the legacy international DashScope
endpoint.  The model defaults to ``qwen3.8-flash`` and can be changed with
``QWEN_MODEL``.  The selected model must support synchronous Chat Completions,
non-thinking mode, and JSON Object structured output.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
BENCHMARK_SCRIPT = PROJECT_ROOT / "run_analysis_llm.py"
DEFAULT_MODEL = "qwen3.8-flash"
LEGACY_INTERNATIONAL_URL = (
    "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
)


def _local_env_value(key: str) -> str:
    """Read one value from .env without importing unrelated provider settings."""
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


def _provider_value(env: dict[str, str], key: str) -> str:
    return env.get(key, "").strip() or _local_env_value(key)


def _default_decision_workers(model: str) -> str:
    """Use model-specific concurrency that remains below provider limits."""
    normalized = model.strip().lower()
    if normalized == "qwen3.7-flash" or normalized.startswith("qwen3.7-flash-"):
        return "30"
    if normalized == "qwen-turbo" or normalized.startswith("qwen-turbo-"):
        return "12"
    return "10"


def _qwen_environment() -> tuple[dict[str, str], str, str, bool]:
    """Build an isolated OpenAI-compatible environment for Qwen Cloud."""
    env = os.environ.copy()
    model = _provider_value(env, "QWEN_MODEL") or DEFAULT_MODEL
    configured_url = _provider_value(env, "QWEN_BASE_URL")
    base_url = configured_url or LEGACY_INTERNATIONAL_URL
    api_key = (
        _provider_value(env, "QWEN_API_KEY")
        or _provider_value(env, "DASHSCOPE_API_KEY")
    )
    if not api_key:
        raise ValueError(
            "Qwen API key missing. Set DASHSCOPE_API_KEY or QWEN_API_KEY "
            "in the environment or in the project .env file."
        )

    # Override every provider-identifying variable so values from another LLM
    # launcher cannot accidentally route this run to a different endpoint.
    env["LLM_ENABLED"] = "1"
    env["LLM_MODEL"] = model
    env["LLM_BASE_URL"] = base_url
    env["LLM_API_KEY"] = api_key

    # Structured, non-thinking output keeps each ecological decision short,
    # deterministic in shape, and compatible with the existing strict parser.
    env["LLM_JSON_MODE"] = "1"
    env["LLM_ENABLE_THINKING"] = "false"
    env.setdefault("LLM_MAX_COMPLETION_TOKENS", "64")
    env.setdefault("LLM_TEMPERATURE", "0.7")
    env.setdefault("LLM_DECISION_MAX_WORKERS", _default_decision_workers(model))
    env.setdefault("LLM_TIMEOUT_SECONDS", "30")
    env.setdefault("LLM_MAX_RETRIES", "10")
    env.setdefault("LLM_RETRY_DELAY", "2.0")
    env.setdefault("LLM_STEP_MAX_RESUMES", "5")
    env.setdefault("LLM_EPISODE_MAX_RESTARTS", "5")
    env.pop("LLM_PROMPT_VARIANT_LOCKED", None)
    return env, model, base_url, not bool(configured_url)


def main() -> None:
    try:
        env, model, base_url, using_legacy_url = _qwen_environment()
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    print(f"\n=== Interactive Qwen Cloud benchmark: {model} ===", flush=True)
    print(f"Endpoint: {base_url}", flush=True)
    print(
        f"Parallel decision calls: {env['LLM_DECISION_MAX_WORKERS']}",
        flush=True,
    )
    print(
        "Recovery: "
        f"{env['LLM_TIMEOUT_SECONDS']}s request timeout; "
        f"up to {env['LLM_STEP_MAX_RESUMES']} resumes from the last valid step; "
        f"up to {env['LLM_EPISODE_MAX_RESTARTS']} automatic episode restarts.",
        flush=True,
    )
    print(
        "Using DASHSCOPE_API_KEY or QWEN_API_KEY from the environment/.env.",
        flush=True,
    )
    if using_legacy_url:
        print(
            "Note: using the legacy international endpoint. Set QWEN_BASE_URL "
            "to the workspace-specific URL shown by Model Studio if required.",
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
