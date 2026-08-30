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
    _baseline_conditioning,
    _project_root,
    _resolve,
    load_benchmark,
    load_config,
)
from .probes import select_probes
from .profile import cue_coverage
from .report import build_report
from .retrieval import HybridRetriever
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


class ConditioningPreparer:
    """Build the shared retrieval, cues, person model, and six conditionings once."""

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
            raise RuntimeError("formal conditioning preparation must build all six conditions")

        provider_cfg = self.config.get("provider", {})
        settings = ProviderSettings.from_env(
            provider_cfg.get("name", "LOCAL"),
            _resolve(self.project_root, provider_cfg.get("env_file", ".env")),
        )
        model_plan = _load_model_plan(self.config, self.project_root)
        expected_profiler = str(model_plan.get("profiler", {}).get("model", ""))
        if not expected_profiler or settings.model != expected_profiler:
            raise RuntimeError(
                "conditioning preparation requires the pre-registered profiler "
                f"{expected_profiler!r}, got {settings.model!r}"
            )
        asset_root_env = str(model_plan.get("asset_root_env", "SIDEPROFILE_ASSET_ROOT"))
        asset_root_value = os.environ.get(asset_root_env, "")
        tokenizer_path = Path(asset_root_value) / "models" / expected_profiler
        if not asset_root_value or not tokenizer_path.is_dir():
            raise RuntimeError(
                f"conditioning preparation requires the frozen profiler tokenizer at {tokenizer_path}"
            )
        from transformers import AutoTokenizer  # noqa: PLC0415

        profiler_tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            local_files_only=True,
            trust_remote_code=False,
        )

        def conditioning_token_count(text: str) -> int:
            return len(profiler_tokenizer.encode(text, add_special_tokens=False))

        def conditioning_token_clip(text: str, limit: int) -> str:
            token_ids = profiler_tokenizer.encode(text, add_special_tokens=False)
            return profiler_tokenizer.decode(
                token_ids[: max(0, limit)], skip_special_tokens=True
            ).rstrip()
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
        examples = load_benchmark(benchmark_path)
        character_ids = list(run_cfg.get("character_ids") or sorted({
            example.character_id for example in examples
        }))
        retrieval_cfg = self.config.get("retrieval", {})
        vector_store_value = retrieval_cfg.get("vector_store")
        retriever = HybridRetriever(
            mode=retrieval_cfg.get("mode", "hybrid"),
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
            embedding_model_key=retrieval_cfg.get(
                "embedding_model_key", "qwen3-embedding-0.6b"
            ),
            reranker_model=(
                os.path.expandvars(retrieval_cfg["reranker_model"])
                if retrieval_cfg.get("reranker_model")
                else None
            ),
            bm25_top_k=int(retrieval_cfg.get("bm25_top_k", 20)),
            dense_top_k=int(retrieval_cfg.get("dense_top_k", 20)),
            final_top_k=int(retrieval_cfg.get("final_top_k", 10)),
        )
        profiling_cfg = self.config.get("profiling", {})
        probes = select_probes(profiling_cfg.get("probe_ids"))
        if len(probes) != 24:
            raise RuntimeError("formal conditioning preparation requires all 24 probes")
        llm = LLMClient(settings, trace_path=output_dir / "trace.jsonl")
        artifacts: list[dict[str, str]] = []
        with CommentCorpus(db_path) as corpus:
            corpus.initialize()
            corpus.add_characters(load_character_catalog(catalog_path))
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
                include_synthetic=False,
                target_tokens=int(profiling_cfg.get("target_tokens", 1000)),
            )
            target_tokens = int(profiling_cfg.get("target_tokens", 1000))
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
                    condition: _baseline_conditioning(
                        llm,
                        condition=condition,
                        result=result,
                        target_tokens=target_tokens,
                        count_tokens=conditioning_token_count,
                        clip_tokens=conditioning_token_clip,
                    )
                    for condition in conditions
                }
                conditioning_path = character_dir / "conditionings.json"
                _write_json(conditioning_path, conditioning)
                conditioning_lengths = {
                    condition: conditioning_token_count(text)
                    for condition, text in conditioning.items()
                }
                controlled = ("raw", "summary", "ours")
                lower_bound = int(target_tokens * 0.95)
                upper_bound = int(target_tokens * 1.05)
                budget_failures = {
                    condition: conditioning_lengths[condition]
                    for condition in controlled
                    if not lower_bound
                    <= conditioning_lengths[condition]
                    <= upper_bound
                }
                if budget_failures:
                    raise RuntimeError(
                        f"conditioning budget failed for {character_id}: {budget_failures}; "
                        f"required {lower_bound}-{upper_bound}"
                    )
                lengths_path = character_dir / "conditioning_lengths.json"
                _write_json(
                    lengths_path,
                    {
                        "target_tokens": target_tokens,
                        "accepted_range": [lower_bound, upper_bound],
                        "estimator": f"{expected_profiler}_tokenizer",
                        "conditions": conditioning_lengths,
                    },
                )
                for path in (
                    person_path,
                    cues_path,
                    retrieval_path,
                    coverage_path,
                    conditioning_path,
                    lengths_path,
                ):
                    artifacts.append(
                        {
                            "path": str(path.relative_to(output_dir)),
                            "sha256": _sha256(path),
                        }
                    )
            corpus_stats = corpus.stats(include_synthetic=False)

        manifest = {
            "schema_version": 1,
            "stage": "conditioning_preparation",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "provider": settings.provider,
            "profiler_model": settings.model,
            "decoding": "provider_default",
            "characters": character_ids,
            "conditions": conditions,
            "probe_ids": [probe.probe_id for probe in probes],
            "retrieval_mode": retriever.effective_mode,
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
    if set(manifest.get("conditions", [])) != set(SUPPORTED_CONDITIONS):
        failures.append("prepared conditionings do not contain all six conditions")
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
        expected_profiler = str(model_plan.get("profiler", {}).get("model", ""))
        if prepared_manifest.get("profiler_model") != expected_profiler:
            raise RuntimeError("prepared conditionings used the wrong profiler model")
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
