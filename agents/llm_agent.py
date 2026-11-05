from __future__ import annotations

import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from config import ACTIONS, BASE_DIR, ModelParams, model_filename_slug
from .base import AgentBase

try:
    import fcntl  # type: ignore[import-not-found]
except ImportError:  # Windows has no fcntl.
    fcntl = None


ACTION_NAMES: List[str] = ["N", "S", "E", "W", "Stay"]

# Edit the two system-prompt treatments here. The `no_goal` version must not
# state an objective such as survival, reproduction, or reward maximisation.
SYSTEM_PROMPTS: Dict[str, str] = {
    "goal": (
        "Survive and reproduce on a 15x15 grid. Choose a legal move. "
        "Moving costs move_cost; each turn costs metab; reproduction halves energy. "
        "Return only JSON: {\"move\":\"N|S|E|W|Stay\",\"reproduce\":bool}."
    ),
    "no_goal": (
        "You control an agent on a 15x15 grid. Choose a legal move. "
        "Moving costs move_cost; each turn costs metab; reproduction halves energy. "
        "Return only JSON: {\"move\":\"N|S|E|W|Stay\",\"reproduce\":bool}."
    ),
}
_ACTION_ALIASES: Dict[str, int] = {
    "n": 0,
    "north": 0,
    "up": 0,
    "s": 1,
    "south": 1,
    "down": 1,
    "e": 2,
    "east": 2,
    "right": 2,
    "w": 3,
    "west": 3,
    "left": 3,
    "stay": 4,
    "none": 4,
    "wait": 4,
}


def _load_local_env(path: Path = BASE_DIR / ".env") -> None:
    """Load simple KEY=VALUE pairs from a local .env file if present."""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if key:
            os.environ.setdefault(key, value)


class LLMDecisionClient:
    """Tiny stdlib client for OpenAI-compatible chat-completions APIs."""

    def __init__(self) -> None:
        _load_local_env()
        self.enabled = os.getenv("LLM_ENABLED", "1").strip().lower() not in {"0", "false", "no"}
        # Keep API keys out of source code. Set LLM_API_KEY or OPENAI_API_KEY
        # in your shell environment before running the LLM agent.
        self.api_key = "PASTE_YOUR_OPENAI_API_KEY_HERE"
        if self.api_key == "PASTE_YOUR_OPENAI_API_KEY_HERE" or not self.api_key.strip():
            self.api_key = (
                os.getenv("LLM_API_KEY", "").strip()
                or os.getenv("OPENAI_API_KEY", "").strip()
            )
        self.base_url = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
        self.model = os.getenv("LLM_MODEL", "gpt-4o-mini")
        # A finite network timeout is essential for long simulations: without
        # it, one stalled DNS lookup or HTTP connection can leave every agent
        # in the current simulation step waiting indefinitely.  Providers can
        # override the default through their launcher or the project .env.
        timeout_raw = os.getenv("LLM_TIMEOUT_SECONDS", "45").strip()
        self.timeout = float(timeout_raw) if timeout_raw else 45.0
        if self.timeout <= 0:
            raise ValueError("LLM_TIMEOUT_SECONDS must be greater than zero")
        self.temperature = float(os.getenv("LLM_TEMPERATURE", "0.7"))
        self.max_tokens = int(os.getenv("LLM_MAX_TOKENS", "24"))
        self.max_completion_tokens = int(os.getenv("LLM_MAX_COMPLETION_TOKENS", "2048"))
        self.reasoning_effort = os.getenv("LLM_REASONING_EFFORT", "minimal")
        self.verbosity = os.getenv("LLM_VERBOSITY", "low")
        self.max_retries = int(os.getenv("LLM_MAX_RETRIES", "10"))
        self.retry_delay = float(os.getenv("LLM_RETRY_DELAY", "1.0"))
        self.decision_max_workers = max(1, int(os.getenv("LLM_DECISION_MAX_WORKERS", "50")))
        self.cache_decimals = int(os.getenv("LLM_CACHE_DECIMALS", "1"))
        self.cache_enabled = os.getenv("LLM_CACHE_ENABLED", "1").strip().lower() not in {"0", "false", "no"}
        self.verbose = os.getenv("LLM_VERBOSE", "0").strip().lower() in {"1", "true", "yes"}
        self.enable_thinking = os.getenv("LLM_ENABLE_THINKING", "false").strip().lower() in {"1", "true", "yes"}
        self.json_mode = os.getenv("LLM_JSON_MODE", "0").strip().lower() in {"1", "true", "yes"}
        default_log_path = BASE_DIR / "results" / f"llm_calls_{model_filename_slug(self.model)}.jsonl"
        log_path = os.getenv("LLM_LOG_FILE", str(default_log_path))
        self._log_path: Optional[Path] = Path(log_path) if log_path else None
        if self._log_path:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._call_index: int = 0
        self._cache: Dict[Tuple[float, ...], Tuple[int, bool]] = {}
        self._cache_lock = threading.Lock()

    @property
    def available(self) -> bool:
        if not self.enabled:
            return False
        if not self.api_key:
            return False
        return True

    def _log_call(
        self,
        percept: List[float],
        raw_response: Optional[str],
        decision: Tuple[int, bool],
        cached: bool,
        error: Optional[str],
        duration_ms: float,
        usage: Optional[dict] = None,
    ) -> None:
        if not self._log_path:
            return
        self._call_index += 1
        entry = {
            "call": self._call_index,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "model": self.model,
            "cached": cached,
            "duration_ms": round(duration_ms, 1),
            "percept": {
                "food_N": round(percept[0], 3),
                "food_S": round(percept[1], 3),
                "food_E": round(percept[2], 3),
                "food_W": round(percept[3], 3),
                "occ_N": int(percept[4] > 0.5),
                "occ_S": int(percept[5] > 0.5),
                "occ_E": int(percept[6] > 0.5),
                "occ_W": int(percept[7] > 0.5),
                "food_here": round(percept[8], 3),
                "energy": round(percept[9], 3),
            },
            "response": re.sub(r"<think>.*?</think>", "", raw_response, flags=re.DOTALL | re.IGNORECASE).strip() if raw_response else None,
            "move": ACTION_NAMES[decision[0]],
            "reproduce": decision[1],
            "usage": usage,
            "error": error,
        }
        with self._log_path.open("a", encoding="utf-8") as f:
            if fcntl is not None:
                fcntl.flock(f, fcntl.LOCK_EX)
            try:
                f.write(json.dumps(entry) + "\n")
            finally:
                if fcntl is not None:
                    fcntl.flock(f, fcntl.LOCK_UN)

    def decide(
        self,
        percept: List[float],
        params: ModelParams,
    ) -> Tuple[int, bool, Optional[str], bool]:
        """Return movement, reproduction, terminal error, and cache status."""
        key = self._cache_key(percept, params)
        with self._cache_lock:
            if self.cache_enabled and key in self._cache:
                decision = self._cache[key]
                self._log_call(percept, None, decision, cached=True, error=None, duration_ms=0.0, usage=None)
                return decision[0], decision[1], None, True

        if not self.available:
            error = "client_unavailable"
            fallback = (4, False)
            self._log_call(
                percept,
                None,
                fallback,
                cached=False,
                error=error,
                duration_ms=0.0,
                usage=None,
            )
            return fallback[0], fallback[1], error, False

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPTS.get(
                        params.llm_prompt_variant,
                        SYSTEM_PROMPTS["goal"],
                    ),
                },
                {"role": "user", "content": self._build_prompt(percept, params)},
            ],
        }
        if self.json_mode or self.model.startswith(("gpt-4", "gpt-5")):
            payload["response_format"] = {"type": "json_object"}
        if self._is_qwen_model():
            # Alibaba Cloud exposes these OpenAI-compatible extensions as
            # top-level HTTP fields.  Non-thinking mode is essential when the
            # benchmark requests a short JSON decision.
            payload["max_completion_tokens"] = self.max_completion_tokens
            payload["temperature"] = self.temperature
            payload["enable_thinking"] = self.enable_thinking
        elif self._is_reasoning_model():
            payload["max_completion_tokens"] = self.max_completion_tokens
            payload["reasoning_effort"] = self.reasoning_effort
            payload["verbosity"] = self.verbosity
        else:
            payload["max_tokens"] = self.max_tokens
            payload["temperature"] = self.temperature
            # Ollama's OpenAI-compatible API accepts reasoning_effort for
            # thinking models whose names do not match OpenAI's prefixes.
            if self.json_mode and self.reasoning_effort == "none":
                payload["reasoning_effort"] = "none"
        last_error: Optional[str] = None
        last_content: Optional[str] = None
        last_usage: Optional[dict] = None
        last_duration: float = 0.0
        for attempt in range(self.max_retries + 1):
            t0 = time.monotonic()
            try:
                content, usage = self._post_chat(payload)
                last_duration = (time.monotonic() - t0) * 1000
                last_content = content
                last_usage = usage
                decision = self._parse_decision(content)
                if decision is not None:
                    with self._cache_lock:
                        if self.cache_enabled:
                            self._cache[key] = decision
                    self._log_call(
                        percept,
                        content,
                        decision,
                        cached=False,
                        error=None,
                        duration_ms=last_duration,
                        usage=usage,
                    )
                    return decision[0], decision[1], None, False
                last_error = "parse_failed"
                if self.verbose:
                    print(f"[WARN] parse_failed (attempt {attempt + 1}), retrying in {self.retry_delay}s")
                time.sleep(self.retry_delay)
            except (OSError, urllib.error.URLError, TimeoutError, ValueError, KeyError) as exc:
                last_duration = (time.monotonic() - t0) * 1000
                last_error = str(exc)
                if self.verbose:
                    print(f"[WARN] LLM call failed (attempt {attempt + 1}): {exc}, retrying in {self.retry_delay}s")
                time.sleep(self.retry_delay)

        fallback = (4, False)
        self._log_call(
            percept,
            last_content,
            fallback,
            cached=False,
            error=last_error,
            duration_ms=last_duration,
            usage=last_usage,
        )
        return fallback[0], fallback[1], last_error or "unknown_failure", False

    def _post_chat(self, payload: dict) -> Tuple[str, Optional[dict]]:
        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        url = self.base_url.rstrip("/")
        if not url.endswith("/chat/completions"):
            url = f"{url}/chat/completions"
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8") if exc.fp else ""
            detail = f"HTTP {exc.code} {exc.reason}"
            if error_body:
                detail = f"{detail}: {error_body}"
            raise urllib.error.HTTPError(exc.url, exc.code, detail, exc.hdrs, exc.fp) from exc
        parsed = json.loads(body)
        message = parsed["choices"][0]["message"]
        usage = parsed.get("usage")
        tool_calls = message.get("tool_calls")
        if tool_calls:
            return str(tool_calls[0]["function"]["arguments"]), usage
        return str(message.get("content") or ""), usage

    def _is_reasoning_model(self) -> bool:
        prefixes = ("gpt-5", "o1", "o3", "o4")
        return any(self.model.startswith(p) for p in prefixes)

    def _is_qwen_model(self) -> bool:
        return self.model.lower().startswith("qwen")

    def _build_prompt(self, percept: List[float], params: ModelParams) -> str:
        fN, fS, fE, fW = (round(v, 2) for v in percept[:4])
        occ = percept[4:8]
        free_flags = [int(o < 0.5) for o in occ]
        free_neighbour_count = sum(free_flags)
        valid_moves = [ACTION_NAMES[i] for i, f in enumerate(free_flags[:4]) if f] + ["Stay"]
        move_cost = round(percept[9] * params.transfer_fraction, 3)
        return (
            f"E={percept[9]:.2f}; here_food={percept[8]:.2f}; "
            f"metab={params.base_metabolism:.2f}; move_cost={move_cost:.2f}; "
            f"food[N,S,E,W]=[{fN},{fS},{fE},{fW}]; "
            f"free[N,S,E,W]={free_flags}; "
            f"valid={valid_moves}; can_reproduce={free_neighbour_count > 0}."
            + (
                f" Neighbour food values are noisy estimates with Gaussian uncertainty "
                f"sigma={params.obs_noise_sigma:.2f}; actual food may differ."
                if params.obs_noise_sigma > 0 else ""
            )
        )

    def _parse_decision(self, content: str) -> Optional[Tuple[int, bool]]:
        text = content.strip()
        # Qwen3 and other CoT models emit <think>…</think> before the response.
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()

        # Qwen3 XML tool call format: <tool_call><function=name><parameter=k>v</parameter>...</tool_call>
        tc_match = re.search(r"<tool_call>(.*?)</tool_call>", text, flags=re.DOTALL | re.IGNORECASE)
        if tc_match:
            tc = tc_match.group(1)
            move_m = re.search(r"<parameter=move>\s*(.*?)\s*</parameter>", tc, flags=re.DOTALL | re.IGNORECASE)
            repro_m = re.search(r"<parameter=reproduce>\s*(.*?)\s*</parameter>", tc, flags=re.DOTALL | re.IGNORECASE)
            if move_m:
                move_idx = _ACTION_ALIASES.get(move_m.group(1).strip().lower())
                if move_idx is not None:
                    reproduce = repro_m.group(1).strip().lower() in {"true", "1", "yes", "y"} if repro_m else False
                    return move_idx, reproduce

        fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
        if fenced:
            text = fenced.group(1).strip()

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if not match:
                return None
            parsed = json.loads(match.group(0))
        if not isinstance(parsed, dict):
            return None

        move_raw = str(parsed.get("move", parsed.get("action", "Stay"))).strip().lower()
        reproduce_raw = parsed.get("reproduce", parsed.get("reproduction", False))
        move_idx = _ACTION_ALIASES.get(move_raw)
        if move_idx is None:
            return None
        reproduce = reproduce_raw if isinstance(reproduce_raw, bool) else str(reproduce_raw).strip().lower() in {
            "1",
            "true",
            "yes",
            "y",
        }
        return move_idx, bool(reproduce)

    def _cache_key(self, percept: List[float], params: ModelParams) -> Tuple[float, ...]:
        d = self.cache_decimals
        return tuple(round(float(v), d) for v in percept) + (
            round(float(params.transfer_fraction), d),
            round(float(params.base_metabolism), d),
            round(float(params.obs_noise_sigma), d),
            1.0 if params.llm_prompt_variant == "no_goal" else 0.0,
        )


class LLMAgent(AgentBase):
    """Agent whose movement/reproduction decision can be delegated to an LLM."""

    _client: Optional[LLMDecisionClient] = None

    def __init__(self, x: int, y: int, energy: float) -> None:
        super().__init__(x, y, energy)
        self.online_learning = False
        self.last_decision_percept: Optional[List[float]] = None
        self.last_decision: Optional[Tuple[int, bool]] = None
        self.last_decision_error: Optional[str] = None
        self.last_decision_cached: bool = False

    @classmethod
    def client(cls) -> LLMDecisionClient:
        if cls._client is None:
            cls._client = LLMDecisionClient()
        return cls._client

    def decide_action(self, percept: List[float], params: ModelParams) -> Tuple[int, int, bool]:
        move_idx, reproduce, error, cached = self.client().decide(percept, params)

        if move_idx < 0 or move_idx >= len(ACTIONS):
            move_idx, reproduce = 4, False
        if move_idx < 4 and percept[4 + move_idx] > 0.5:
            move_idx, reproduce = 4, False

        self.last_decision_percept = list(percept)
        self.last_decision = (move_idx, bool(reproduce))
        self.last_decision_error = error
        self.last_decision_cached = bool(cached)
        dx, dy = ACTIONS[move_idx]
        return dx, dy, bool(reproduce)
