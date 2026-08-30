from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def load_dotenv(path: str | Path | None = None) -> Path | None:
    candidates = []
    if path:
        candidates.append(Path(path))
    configured = os.environ.get("SIDEPROFILE_ENV_FILE")
    if configured:
        candidates.append(Path(configured))
    candidates.extend([Path.cwd() / ".env", Path.cwd().parent / ".env"])
    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.expanduser().resolve()
        if candidate in seen or not candidate.is_file():
            continue
        seen.add(candidate)
        for line in candidate.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and value and key not in os.environ:
                os.environ[key] = value
        return candidate
    return None


@dataclass(frozen=True)
class ProviderSettings:
    provider: str
    api_key: str
    base_url: str | None
    model: str

    @classmethod
    def from_env(cls, provider: str = "GPT", env_file: str | Path | None = None) -> "ProviderSettings":
        load_dotenv(env_file)
        prefix = provider.upper()
        api_key = os.environ.get(f"{prefix}_API_KEY", "")
        base_url = os.environ.get(f"{prefix}_BASE_URL") or None
        model = os.environ.get(f"{prefix}_MODEL", "")
        if not api_key or not model:
            raise RuntimeError(
                f"incomplete {prefix} configuration: set {prefix}_API_KEY and {prefix}_MODEL in .env"
            )
        return cls(provider=prefix, api_key=api_key, base_url=base_url, model=model)


@dataclass
class Usage:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


def parse_json_object(text: str) -> dict:
    value = text.strip()
    if value.startswith("```"):
        value = value.split("```", 2)[1]
        if value.lstrip().startswith("json"):
            value = value.lstrip()[4:]
    start, end = value.find("{"), value.rfind("}")
    if start >= 0 and end > start:
        value = value[start : end + 1]
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("LLM output is not a JSON object")
    return parsed


class LLMClient:
    def __init__(
        self,
        settings: ProviderSettings,
        *,
        trace_path: str | Path | None = None,
    ) -> None:
        from openai import OpenAI

        self.settings = settings
        self.client = OpenAI(api_key=settings.api_key, base_url=settings.base_url)
        self.usage = Usage()
        self._state_lock = threading.Lock()
        self.trace_path = Path(trace_path) if trace_path else None
        if self.trace_path:
            self.trace_path.parent.mkdir(parents=True, exist_ok=True)

    def _charge(self) -> None:
        with self._state_lock:
            self.usage.calls += 1

    def _write_trace(self, record: dict) -> None:
        if not self.trace_path:
            return
        with self._state_lock:
            with self.trace_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def chat(
        self,
        *,
        system: str,
        user: str,
        agent: str,
        json_mode: bool = False,
    ) -> str:
        self._charge()
        kwargs: dict[str, Any] = {
            "model": self.settings.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        response = self.client.chat.completions.create(**kwargs)
        text = response.choices[0].message.content or ""
        usage = getattr(response, "usage", None)
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        with self._state_lock:
            self.usage.prompt_tokens += prompt_tokens
            self.usage.completion_tokens += completion_tokens
        self._write_trace(
            {
                "timestamp": time.time(),
                "agent": agent,
                "model": self.settings.model,
                "request": {"system": system, "user": user},
                "response": text,
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                },
            }
        )
        return text

    def chat_json(self, **kwargs: Any) -> dict:
        kwargs["json_mode"] = True
        return parse_json_object(self.chat(**kwargs))

    def probe(self) -> dict:
        payload = self.chat_json(
            system="You are a connectivity test. Return JSON only.",
            user='Return {"status":"ok","purpose":"sideprofile-smoke"}.',
            agent="provider_probe",
        )
        if payload.get("status") != "ok":
            raise RuntimeError(f"unexpected provider probe response: {payload}")
        return {
            "status": "ok",
            "provider": self.settings.provider,
            "model": self.settings.model,
            "usage": self.usage.__dict__,
        }


class MockLLMClient:
    """Small deterministic test double implementing the LLMClient surface."""

    def __init__(self) -> None:
        self.settings = ProviderSettings("MOCK", "not-used", None, "mock")
        self.usage = Usage()

    def chat_json(self, *, agent: str, **_: Any) -> dict:
        self.usage.calls += 1
        if agent.startswith("cue:"):
            probe_id = agent.split(":", 1)[1]
            return {
                "cues": [
                    {
                        "cue": "Unexpected changes trigger an attempt to restore structure.",
                        "context": "routine disruption",
                        "cue_type": "behavioral_pattern",
                        "directness": "behavior_based",
                        "support": ["demo_c01", "demo_c02"],
                        "counterevidence": ["demo_c03"],
                        "confidence": 0.82,
                    }
                ],
                "probe_id": probe_id,
            }
        if agent == "person_model":
            from .schema import PERSON_MODEL_SECTIONS

            return {
                "sections": {
                    section: (["IF routine changes unexpectedly, attempts to restore structure."] if section == "situation_behavior_signatures" else [])
                    for section in PERSON_MODEL_SECTIONS
                }
            }
        if agent == "judge":
            return {"score": 0.75, "rationale": "Behavior follows the supplied model."}
        return {}

    def chat(self, *, agent: str, **_: Any) -> str:
        self.usage.calls += 1
        if agent == "actor":
            return "I will first restore the agreed plan, unless someone close genuinely needs help."
        return "mock response"
