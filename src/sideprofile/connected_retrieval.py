from __future__ import annotations

import hashlib
import json
import math
import ssl
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import certifi

from .llm import load_dotenv


JsonTransport = Callable[[str, str, dict[str, Any]], tuple[int, dict[str, Any]]]


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _embedding_endpoint(value: str) -> str:
    endpoint = value.rstrip("/")
    if endpoint.endswith("/embeddings"):
        return endpoint
    if endpoint.endswith("/openai/v1"):
        return endpoint + "/embeddings"
    return endpoint + "/openai/v1/embeddings"


def _rerank_endpoint(value: str) -> str:
    endpoint = value.rstrip("/")
    if endpoint.endswith("/rerank"):
        return endpoint
    return endpoint + "/v2/rerank"


def _post_bearer_json(
    url: str,
    api_key: str,
    payload: dict[str, Any],
) -> tuple[int, dict[str, Any]]:
    request = Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    context = ssl.create_default_context(cafile=certifi.where())
    try:
        with urlopen(request, timeout=120, context=context) as response:
            status = int(response.status)
            body = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"connected retrieval HTTP {exc.code}: {body}") from exc
    except URLError as exc:
        raise RuntimeError(f"connected retrieval request failed: {exc.reason}") from exc
    if not isinstance(body, dict):
        raise RuntimeError("connected retrieval response is not a JSON object")
    return status, body


@dataclass(frozen=True)
class EmbeddingSettings:
    api_key: str
    endpoint: str
    model: str
    provider: str = "GPT"

    @classmethod
    def from_env(cls, env_file: str | Path | None = None) -> "EmbeddingSettings":
        import os

        load_dotenv(env_file)
        api_key = os.environ.get("GPT_API_KEY", "")
        endpoint = os.environ.get("GPT_EMBEDDING_END_POINT", "")
        model = os.environ.get("GPT_EMBEDDING_MODEL", "")
        if not api_key or not endpoint or not model:
            raise RuntimeError(
                "incomplete embedding configuration: set GPT_API_KEY, "
                "GPT_EMBEDDING_END_POINT, and GPT_EMBEDDING_MODEL in .env"
            )
        return cls(api_key=api_key, endpoint=_embedding_endpoint(endpoint), model=model)


@dataclass(frozen=True)
class RerankerSettings:
    api_key: str
    endpoint: str
    model: str
    provider: str = "COHERE"

    @classmethod
    def from_env(cls, env_file: str | Path | None = None) -> "RerankerSettings":
        import os

        load_dotenv(env_file)
        api_key = os.environ.get("COHERE_API_KEY", "")
        endpoint = os.environ.get("COHERE_END_POINT", "")
        model = os.environ.get("COHERE_RERANK_MODEL", "")
        if not api_key or not endpoint or not model:
            raise RuntimeError(
                "incomplete reranker configuration: set COHERE_API_KEY, "
                "COHERE_END_POINT, and COHERE_RERANK_MODEL in .env"
            )
        return cls(api_key=api_key, endpoint=_rerank_endpoint(endpoint), model=model)


class _TraceWriter:
    def __init__(self, path: str | Path | None) -> None:
        self.path = Path(path) if path else None
        self._lock = threading.Lock()
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: dict[str, Any]) -> None:
        if not self.path:
            return
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")


class OpenAIEmbeddingClient:
    def __init__(
        self,
        settings: EmbeddingSettings,
        *,
        trace_path: str | Path | None = None,
        transport: JsonTransport = _post_bearer_json,
    ) -> None:
        self.settings = settings
        self.transport = transport
        self.trace = _TraceWriter(trace_path)
        self.calls = 0
        self.prompt_tokens = 0
        self.total_tokens = 0
        self.returned_models: set[str] = set()

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        status, response = self.transport(
            self.settings.endpoint,
            self.settings.api_key,
            {"model": self.settings.model, "input": texts},
        )
        if status < 200 or status >= 300:
            raise RuntimeError(f"embedding request returned HTTP {status}")
        data = response.get("data")
        if not isinstance(data, list) or len(data) != len(texts):
            raise RuntimeError("embedding endpoint did not return one vector per input")
        indexed: dict[int, list[float]] = {}
        for fallback_index, item in enumerate(data):
            if not isinstance(item, dict):
                raise RuntimeError("embedding response item is not an object")
            index = int(item.get("index", fallback_index))
            vector = item.get("embedding")
            if (
                index in indexed
                or index < 0
                or index >= len(texts)
                or not isinstance(vector, list)
                or not vector
            ):
                raise RuntimeError("embedding response contains invalid indices or vectors")
            values = [float(value) for value in vector]
            if not all(math.isfinite(value) for value in values):
                raise RuntimeError("embedding response contains a non-finite value")
            indexed[index] = values
        if set(indexed) != set(range(len(texts))):
            raise RuntimeError("embedding endpoint did not return one vector per input")
        dimensions = {len(vector) for vector in indexed.values()}
        if len(dimensions) != 1:
            raise RuntimeError("embedding response dimensions differ")
        usage = response.get("usage") if isinstance(response.get("usage"), dict) else {}
        returned_model = str(response.get("model") or self.settings.model)
        self.calls += 1
        self.prompt_tokens += int(usage.get("prompt_tokens", 0) or 0)
        self.total_tokens += int(usage.get("total_tokens", 0) or 0)
        self.returned_models.add(returned_model)
        self.trace.write(
            {
                "timestamp": time.time(),
                "provider": self.settings.provider,
                "model": self.settings.model,
                "returned_model": returned_model,
                "input_count": len(texts),
                "input_sha256": [_sha256(text) for text in texts],
                "dimension": next(iter(dimensions)),
                "usage": usage,
            }
        )
        return [indexed[index] for index in range(len(texts))]

    @property
    def usage(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "total_tokens": self.total_tokens,
            "returned_models": sorted(self.returned_models),
        }


class CohereReranker:
    def __init__(
        self,
        settings: RerankerSettings,
        *,
        trace_path: str | Path | None = None,
        transport: JsonTransport = _post_bearer_json,
    ) -> None:
        self.settings = settings
        self.transport = transport
        self.trace = _TraceWriter(trace_path)
        self.calls = 0
        self.search_units = 0

    def scores(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []
        status, response = self.transport(
            self.settings.endpoint,
            self.settings.api_key,
            {
                "model": self.settings.model,
                "query": query,
                "documents": documents,
            },
        )
        if status < 200 or status >= 300:
            raise RuntimeError(f"reranker request returned HTTP {status}")
        results = response.get("results")
        if not isinstance(results, list) or len(results) != len(documents):
            raise RuntimeError("reranker must return one score per submitted document")
        indexed: dict[int, float] = {}
        result_indices: list[int] = []
        result_scores: list[float] = []
        for item in results:
            if not isinstance(item, dict):
                raise RuntimeError("reranker result is not an object")
            index = int(item.get("index", -1))
            score = float(item.get("relevance_score", float("nan")))
            if (
                index in indexed
                or index < 0
                or index >= len(documents)
                or not math.isfinite(score)
            ):
                raise RuntimeError("reranker returned invalid indices or scores")
            indexed[index] = score
            result_indices.append(index)
            result_scores.append(score)
        if set(indexed) != set(range(len(documents))):
            raise RuntimeError("reranker must return one score per submitted document")
        meta = response.get("meta") if isinstance(response.get("meta"), dict) else {}
        billed = meta.get("billed_units") if isinstance(meta.get("billed_units"), dict) else {}
        self.calls += 1
        self.search_units += int(billed.get("search_units", 0) or 0)
        self.trace.write(
            {
                "timestamp": time.time(),
                "provider": self.settings.provider,
                "model": self.settings.model,
                "query_sha256": _sha256(query),
                "document_sha256": [_sha256(document) for document in documents],
                "result_indices": result_indices,
                "relevance_scores": result_scores,
                "meta": meta,
            }
        )
        return [indexed[index] for index in range(len(documents))]

    @property
    def usage(self) -> dict[str, int]:
        return {"calls": self.calls, "search_units": self.search_units}
