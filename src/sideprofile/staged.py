from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .anonymize import anonymize_text
from .bundle import verify_bundle_manifest
from .corpus import CommentCorpus, load_character_catalog
from .llm import LLMClient, ProviderSettings
from .pipeline import (
    SUPPORTED_CONDITIONS,
    ProfileBuilder,
    _condition_payload,
    _project_root,
    _resolve,
    load_benchmark,
    load_config,
)
from .probes import select_probes
from .profile import cue_coverage
from .report import build_report
from .retrieval import VectorRerankRetriever, build_retriever
from .roleplay import generate_response
from .schema import GenerationRecord, PersonModel


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _validate_conditions(conditions: list[str]) -> None:
    unsupported = [item for item in conditions if item not in SUPPORTED_CONDITIONS]
    if unsupported:
        raise ValueError("unsupported conditions: " + ", ".join(unsupported))


def _require_frozen_bundle(config: dict[str, Any], project_root: Path) -> dict[str, Any]:
    data = config["data"]
    manifest = data.get("bundle_manifest")
    if not manifest:
        raise RuntimeError("research stage requires data.bundle_manifest")
    result = verify_bundle_manifest(project_root=project_root, manifest_path=manifest)
    if not result["ok"]:
        raise RuntimeError(
            "frozen bundle verification failed: " + "; ".join(result["failures"])
        )
    result["manifest_sha256"] = _sha256(_resolve(project_root, manifest))
    return result


def _load_model_plan(config: dict[str, Any], project_root: Path) -> dict[str, Any]:
    registry = config.get("data", {}).get("model_registry", "offline/models.yaml")
    path = _resolve(project_root, registry)
    if not path.is_file():
        raise RuntimeError(f"missing frozen model registry: {path}")
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"invalid frozen model registry: {path}")
    return value


def _panel_key(run_cfg: dict[str, Any]) -> str:
    name = str(run_cfg.get("name", "")).strip().lower().replace("-", "_")
    if not name:
        raise RuntimeError("run.name is required for staged execution")
    return name


def _profiler_spec(model_plan: dict[str, Any]) -> tuple[str, str, str]:
    spec = model_plan.get("profiler") or {}
    provider = str(spec.get("provider") or "").strip().upper()
    model = str(spec.get("model") or "").strip()
    location = str(spec.get("execution_location") or "").strip()
    if not provider or not model or not location:
        raise RuntimeError(
            "model registry profiler requires provider, model, and execution_location"
        )
    return provider, model, location


def _retrieval_service_spec(model_plan: dict[str, Any]) -> dict[str, str]:
    preparation = model_plan.get("retrieval_preparation") or {}
    values: dict[str, str] = {}
    for role in ("embedding", "reranker"):
        spec = preparation.get(role) or {}
        provider = str(spec.get("provider") or "").strip().upper()
        model = str(spec.get("model") or "").strip()
        location = str(spec.get("execution_location") or "").strip()
        if not provider or not model or location != "connected_preparation":
            raise RuntimeError(
                f"model registry {role} requires provider, model, and "
                "execution_location=connected_preparation"
            )
        values[f"{role}_provider"] = provider
        values[f"{role}_model"] = model
    return values


class ConditioningPreparer:
    """Build shared retrieval, cues, person model, and five condition payloads once."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.config_path = Path(config["_config_path"])
        self.project_root = _project_root(self.config_path)
        self.output_dir: Path | None = None

    @classmethod
    def from_file(cls, path: str | Path) -> "ConditioningPreparer":
        return cls(load_config(path))

    def run(self) -> Path:
        bundle = _require_frozen_bundle(self.config, self.project_root)
        data_cfg = self.config["data"]
        run_cfg = self.config["run"]
        conditions = list(run_cfg.get("conditions", SUPPORTED_CONDITIONS))
        _validate_conditions(conditions)
        if set(conditions) != set(SUPPORTED_CONDITIONS):
            raise RuntimeError("formal conditioning preparation must build all five conditions")

        model_plan = _load_model_plan(self.config, self.project_root)
        expected_provider, expected_profiler, execution_location = _profiler_spec(model_plan)
        retrieval_services = _retrieval_service_spec(model_plan)
        if execution_location != "connected_preparation":
            raise RuntimeError(
                "conditioning preparation requires profiler execution_location "
                "'connected_preparation'"
            )
        provider_cfg = self.config.get("provider", {})
        env_file = _resolve(self.project_root, provider_cfg.get("env_file", ".env"))
        settings = ProviderSettings.from_env(
            expected_provider,
            env_file,
        )
        if settings.model != expected_profiler:
            raise RuntimeError(
                "conditioning preparation requires the pre-registered profiler "
                f"{expected_provider}/{expected_profiler!r}, got "
                f"{settings.provider}/{settings.model!r}"
            )
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        name = str(run_cfg.get("name", "conditioning"))
        output_root = _resolve(
            self.project_root, run_cfg.get("preparation_output_dir", "prepared")
        )
        output_dir = output_root / f"{name}-{stamp}"
        output_dir.mkdir(parents=True, exist_ok=False)
        self.output_dir = output_dir
        _write_json(output_dir / "status.json", {"status": "running"})
        shutil.copy2(self.config_path, output_dir / "config.yaml")

        db_path = _resolve(self.project_root, data_cfg["corpus_db"])
        catalog_path = _resolve(self.project_root, data_cfg["catalog"])
        benchmark_path = _resolve(self.project_root, data_cfg["benchmark"])
        catalog = {
            spec.character_id: spec for spec in load_character_catalog(catalog_path)
        }
        examples = load_benchmark(benchmark_path)
        character_ids = list(run_cfg.get("character_ids") or sorted({
            example.character_id for example in examples
        }))
        missing_catalog = [item for item in character_ids if item not in catalog]
        missing_gold = [
            item
            for item in character_ids
            if item in catalog and "gold" in conditions and not catalog[item].gold_profile
        ]
        if missing_catalog or missing_gold:
            raise RuntimeError(
                "frozen catalog preflight failed: "
                f"missing_characters={missing_catalog}; missing_gold={missing_gold}"
            )
        retrieval_cfg = self.config.get("retrieval", {})
        for key, expected in retrieval_services.items():
            observed = str(retrieval_cfg.get(key, ""))
            if key.endswith("_provider"):
                observed = observed.upper()
            if observed != expected:
                raise RuntimeError(
                    f"conditioning preparation requires {key}={expected!r}, got {observed!r}"
                )
        vector_store_value = retrieval_cfg.get("vector_store")
        retriever = build_retriever(
            mode=str(retrieval_cfg.get("mode", "vector_rerank")),
            vector_store=(
                str(
                    _resolve(
                        self.project_root,
                        os.path.expandvars(str(vector_store_value)),
                    )
                )
                if vector_store_value
                else None
            ),
            embedding_model_key=str(
                retrieval_cfg.get("embedding_model", "text-embedding-3-small")
            ),
            reranker_model=str(
                retrieval_cfg.get("reranker_model", "Cohere-rerank-v4.0-pro")
            ),
            env_file=str(env_file),
            rerank_trace_path=str(output_dir / "rerank.trace.jsonl"),
            candidate_top_k=int(retrieval_cfg.get("candidate_top_k", 20)),
            final_top_k=int(retrieval_cfg.get("final_top_k", 10)),
        )
        if not isinstance(retriever, VectorRerankRetriever):
            raise RuntimeError("formal conditioning preparation requires vector_rerank")
        profiling_cfg = self.config.get("profiling", {})
        probes = select_probes(profiling_cfg.get("probe_ids"))
        if len(probes) != 24:
            raise RuntimeError("formal conditioning preparation requires all 24 probes")
        llm = LLMClient(settings, trace_path=output_dir / "trace.jsonl")
        artifacts: list[dict[str, str]] = []
        with CommentCorpus(db_path, read_only=True) as corpus:
            corpus.initialize()
            coverage = data_cfg["coverage"]
            audit = corpus.validate_targets(
                min_comments=int(coverage["min_comments"]),
                min_platforms=int(coverage["min_platforms"]),
                min_authors=int(coverage["min_authors"]),
                include_synthetic=False,
                character_ids=character_ids,
            )
            if not audit["ready"]:
                failures = [
                    f"{row['character_id']} ({', '.join(row['failures'])})"
                    for row in audit["characters"]
                    if not row["ready"]
                ]
                raise RuntimeError("corpus coverage gate failed: " + "; ".join(failures))
            builder = ProfileBuilder(
                corpus,
                llm,
                retriever,
                characters=catalog,
                include_synthetic=False,
            )
            for character_id in character_ids:
                result = builder.build(character_id, probes)
                character_dir = output_dir / "characters" / character_id
                character_dir.mkdir(parents=True)
                person_path = character_dir / "person_model.json"
                person_path.write_text(
                    result.person_model.model_dump_json(indent=2) + "\n", encoding="utf-8"
                )
                cues_path = character_dir / "cues.jsonl"
                with cues_path.open("w", encoding="utf-8") as handle:
                    for cue in result.cues:
                        handle.write(cue.model_dump_json() + "\n")
                retrieval_path = character_dir / "retrieval.json"
                _write_json(retrieval_path, result.retrieval)
                coverage_path = character_dir / "coverage.json"
                _write_json(coverage_path, cue_coverage(result.cues))
                conditioning = {
                    condition: _condition_payload(
                        llm,
                        condition=condition,
                        result=result,
                    )
                    for condition in conditions
                }
                conditioning_path = character_dir / "conditionings.json"
                _write_json(conditioning_path, conditioning)
                for path in (
                    person_path,
                    cues_path,
                    retrieval_path,
                    coverage_path,
                    conditioning_path,
                ):
                    artifacts.append(
                        {
                            "path": str(path.relative_to(output_dir)),
                            "sha256": _sha256(path),
                        }
                    )
            corpus_stats = corpus.stats(include_synthetic=False)

        trace_path = output_dir / "trace.jsonl"
        if trace_path.is_file():
            artifacts.append(
                {
                    "path": str(trace_path.relative_to(output_dir)),
                    "sha256": _sha256(trace_path),
                }
            )
        rerank_trace_path = output_dir / "rerank.trace.jsonl"
        if rerank_trace_path.is_file():
            artifacts.append(
                {
                    "path": str(rerank_trace_path.relative_to(output_dir)),
                    "sha256": _sha256(rerank_trace_path),
                }
            )

        manifest = {
            "schema_version": 1,
            "stage": "conditioning_preparation",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "provider": settings.provider,
            "profiler_provider": settings.provider,
            "profiler_model": settings.model,
            "decoding": "provider_default",
            "characters": character_ids,
            "conditions": conditions,
            "probe_ids": [probe.probe_id for probe in probes],
            "comment_processing": "isolated_per_probe_top10",
            "condition_aggregation": {
                "personality": "per_probe_observations_then_aggregate",
                "summary": "per_probe_summaries_then_aggregate",
                "ours": "per_probe_cues_then_person_model",
            },
            "retrieval_mode": retriever.effective_mode,
            "embedding_provider": retrieval_services["embedding_provider"],
            "embedding_model": retriever.embedding_model_key,
            "reranker_provider": retriever.reranker.settings.provider,
            "reranker_model": retriever.reranker.settings.model,
            "candidate_top_k": retriever.candidate_top_k,
            "final_top_k": retriever.final_top_k,
            "reranker_usage": retriever.reranker.usage,
            "corpus_stats": corpus_stats,
            "coverage_audit": audit,
            "input_bundle": bundle["manifest"],
            "input_bundle_sha256": bundle["manifest_sha256"],
            "llm_usage": llm.usage.__dict__,
            "research_valid": True,
            "artifacts": artifacts,
        }
        _write_json(output_dir / "manifest.json", manifest)
        _write_json(output_dir / "status.json", {"status": "completed"})
        _write_json(output_root / f"{name}-latest.json", {"prepared_dir": str(output_dir)})
        return output_dir


def verify_prepared(path: str | Path) -> dict[str, Any]:
    root = Path(path).resolve()
    failures: list[str] = []
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        return {"ok": False, "failures": [f"missing {manifest_path}"]}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("stage") != "conditioning_preparation":
        failures.append("wrong prepared stage")
    if manifest.get("research_valid") is not True:
        failures.append("prepared conditionings are not research_valid")
    if not manifest.get("profiler_provider") or not manifest.get("profiler_model"):
        failures.append("prepared conditionings do not identify the fixed profiler")
    if manifest.get("retrieval_mode") != "openai_exact_vector_recall+cohere_rerank":
        failures.append("prepared conditionings used the wrong retrieval mode")
    for field in (
        "embedding_provider",
        "embedding_model",
        "reranker_provider",
        "reranker_model",
    ):
        if not manifest.get(field):
            failures.append(f"prepared conditionings do not identify {field}")
    if manifest.get("candidate_top_k") != 20 or manifest.get("final_top_k") != 10:
        failures.append("prepared conditionings used the wrong retrieval Top-K contract")
    if set(manifest.get("conditions", [])) != set(SUPPORTED_CONDITIONS):
        failures.append("prepared conditionings do not contain all five conditions")
    if manifest.get("comment_processing") != "isolated_per_probe_top10":
        failures.append("prepared comments were not processed as isolated per-probe Top-10 sets")
    if manifest.get("condition_aggregation") != {
        "personality": "per_probe_observations_then_aggregate",
        "summary": "per_probe_summaries_then_aggregate",
        "ours": "per_probe_cues_then_person_model",
    }:
        failures.append("prepared condition aggregation contract differs")
    for item in manifest.get("artifacts", []):
        artifact = root / item["path"]
        if not artifact.is_file():
            failures.append(f"missing artifact: {item['path']}")
        elif _sha256(artifact) != item["sha256"]:
            failures.append(f"checksum mismatch: {item['path']}")
    return {"ok": not failures, "failures": failures, "manifest": manifest}


class ActorRunner:
    """Run one actor against immutable shared conditionings without rebuilding profiles."""

    def __init__(self, config: dict[str, Any], prepared_dir: str | Path) -> None:
        self.config = config
        self.config_path = Path(config["_config_path"])
        self.project_root = _project_root(self.config_path)
        self.prepared_dir = Path(prepared_dir).resolve()
        self.run_dir: Path | None = None

    @classmethod
    def from_file(cls, path: str | Path, prepared_dir: str | Path) -> "ActorRunner":
        return cls(load_config(path), prepared_dir)

    def run(self) -> Path:
        bundle = _require_frozen_bundle(self.config, self.project_root)
        prepared = verify_prepared(self.prepared_dir)
        if not prepared["ok"]:
            raise RuntimeError(
                "prepared conditioning verification failed: "
                + "; ".join(prepared["failures"])
            )
        prepared_manifest = prepared["manifest"]
        if (
            prepared_manifest.get("input_bundle") != bundle["manifest"]
            or prepared_manifest.get("input_bundle_sha256") != bundle["manifest_sha256"]
        ):
            raise RuntimeError("prepared conditionings were built from a different frozen input bundle")
        data_cfg = self.config["data"]
        run_cfg = self.config["run"]
        conditions = list(run_cfg.get("conditions", SUPPORTED_CONDITIONS))
        _validate_conditions(conditions)
        if conditions != prepared_manifest["conditions"]:
            raise RuntimeError("actor conditions differ from prepared conditions")
        character_ids = list(run_cfg.get("character_ids", prepared_manifest["characters"]))
        if character_ids != prepared_manifest["characters"]:
            raise RuntimeError("actor character order differs from prepared conditionings")

        provider_cfg = self.config.get("provider", {})
        settings = ProviderSettings.from_env(
            provider_cfg.get("name", "LOCAL"),
            _resolve(self.project_root, provider_cfg.get("env_file", ".env")),
        )
        model_plan = _load_model_plan(self.config, self.project_root)
        panel_key = _panel_key(run_cfg)
        allowed_actors = list(model_plan.get("actor_matrix", {}).get(panel_key, []))
        if not allowed_actors or settings.model not in allowed_actors:
            raise RuntimeError(
                f"actor {settings.model!r} is not pre-registered for {panel_key}: "
                + ", ".join(allowed_actors)
            )
        expected_provider, expected_profiler, _ = _profiler_spec(model_plan)
        if (
            prepared_manifest.get("profiler_provider") != expected_provider
            or prepared_manifest.get("profiler_model") != expected_profiler
        ):
            raise RuntimeError("prepared conditionings used the wrong profiler model")
        retrieval_services = _retrieval_service_spec(model_plan)
        if any(
            prepared_manifest.get(key) != expected
            for key, expected in retrieval_services.items()
        ):
            raise RuntimeError("prepared conditionings used the wrong retrieval services")
        safe_model = re.sub(r"[^A-Za-z0-9_.-]+", "-", settings.model).strip("-")[-80:]
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_id = f"{run_cfg.get('name', 'actor')}-{safe_model}-{stamp}"
        output_root = _resolve(self.project_root, run_cfg.get("output_dir", "runs"))
        run_dir = output_root / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        self.run_dir = run_dir
        _write_json(run_dir / "status.json", {"status": "running", "run_id": run_id})
        shutil.copy2(self.config_path, run_dir / "config.yaml")

        benchmark_path = _resolve(self.project_root, data_cfg["benchmark"])
        examples = load_benchmark(benchmark_path)
        catalog = {
            spec.character_id: spec
            for spec in load_character_catalog(
                _resolve(self.project_root, data_cfg["catalog"])
            )
        }
        replicates = int(run_cfg.get("replicates", 3))
        if replicates < 1:
            raise ValueError("run.replicates must be at least 1")
        llm = LLMClient(settings, trace_path=run_dir / "trace.jsonl")
        generation_path = run_dir / "generations.jsonl"
        for character_id in character_ids:
            spec = catalog[character_id]
            character_dir = self.prepared_dir / "characters" / character_id
            person_model = PersonModel.model_validate_json(
                (character_dir / "person_model.json").read_text(encoding="utf-8")
            )
            conditioning = json.loads(
                (character_dir / "conditionings.json").read_text(encoding="utf-8")
            )
            for example in [item for item in examples if item.character_id == character_id]:
                for condition in conditions:
                    for replicate in range(1, replicates + 1):
                        output = generate_response(
                            llm,
                            example=example,
                            model=person_model if condition == "ours" else None,
                            condition=condition,
                            conditioning_text=conditioning[condition],
                        )
                        output = anonymize_text(output, spec)
                        record = GenerationRecord(
                            run_id=run_id,
                            example_id=example.example_id,
                            character_id=character_id,
                            anonymous_id=person_model.anonymous_id,
                            condition=condition,
                            replicate=replicate,
                            query=example.query,
                            context=example.context,
                            output=output,
                            actor_model=settings.model,
                        )
                        with generation_path.open("a", encoding="utf-8") as handle:
                            handle.write(record.model_dump_json() + "\n")
        manifest = {
            "run_id": run_id,
            "stage": "actor_generation",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "provider": settings.provider,
            "model": settings.model,
            "decoding": "provider_default",
            "characters": character_ids,
            "conditions": conditions,
            "replicates": replicates,
            "probe_ids": prepared_manifest["probe_ids"],
            "retrieval_mode": prepared_manifest["retrieval_mode"],
            "corpus_stats": prepared_manifest["corpus_stats"],
            "prepared_dir": str(self.prepared_dir),
            "prepared_manifest_sha256": _sha256(self.prepared_dir / "manifest.json"),
            "official_evaluation_status": "pending",
            "llm_usage": llm.usage.__dict__,
            "research_valid": True,
        }
        _write_json(run_dir / "manifest.json", manifest)
        _write_json(run_dir / "status.json", {"status": "completed", "run_id": run_id})
        build_report(run_dir)
        return run_dir
