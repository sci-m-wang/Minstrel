from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .anonymize import anonymize_text, find_identity_leaks
from .bundle import verify_bundle_manifest
from .corpus import CommentCorpus, load_character_catalog
from .llm import LLMClient, MockLLMClient, ProviderSettings
from .probes import PROBES_BY_ID, select_probes
from .profile import (
    build_person_model,
    cue_coverage,
    extract_cues,
    render_person_model,
)
from .report import build_report
from .retrieval import Retriever, build_retriever
from .roleplay import generate_response, judge_response
from .schema import BenchmarkExample, CharacterSpec, Cue, GenerationRecord, PersonModel, Probe

SUPPORTED_CONDITIONS = ["none", "personality", "summary", "gold", "ours"]


@dataclass
class ProfileResult:
    character: CharacterSpec
    person_model: PersonModel
    cues: list[Cue]
    retrieval: dict[str, list[dict]]
    retrieval_mode: str


class ProfileBuilder:
    def __init__(
        self,
        corpus: CommentCorpus,
        llm: LLMClient | MockLLMClient,
        retriever: Retriever,
        *,
        characters: dict[str, CharacterSpec] | None = None,
        include_synthetic: bool = False,
    ) -> None:
        self.corpus = corpus
        self.llm = llm
        self.retriever = retriever
        self.characters = characters
        self.include_synthetic = include_synthetic

    def build(self, character_id: str, probes: list[Probe]) -> ProfileResult:
        if self.characters is not None:
            try:
                spec = self.characters[character_id]
            except KeyError as exc:
                raise RuntimeError(
                    f"frozen catalog is missing character {character_id}"
                ) from exc
        else:
            spec = self.corpus.get_character(character_id)
        comments = self.corpus.comments_for(
            character_id, include_synthetic=self.include_synthetic
        )
        if not comments:
            suffix = " (synthetic rows are excluded)" if not self.include_synthetic else ""
            raise RuntimeError(f"no comments available for {character_id}{suffix}")
        cues: list[Cue] = []
        retrieval: dict[str, list[dict]] = {}
        for probe in probes:
            retrieved = self.retriever.retrieve(probe, comments, spec)
            for item in retrieved:
                leaks = find_identity_leaks(item.text, spec)
                if leaks:
                    raise RuntimeError(
                        f"identity leak after anonymization for {item.comment_id}: {leaks}"
                    )
            retrieval[probe.probe_id] = [item.model_dump() for item in retrieved]
            cues.extend(
                extract_cues(
                    self.llm,
                    spec=spec,
                    probe=probe,
                    retrieved=retrieved,
                )
            )
        person_model = build_person_model(
            self.llm,
            spec=spec,
            cues=cues,
        )
        rendered = render_person_model(person_model)
        leaks = find_identity_leaks(rendered, spec)
        if leaks:
            raise RuntimeError(f"person model leaked private identity strings: {leaks}")
        return ProfileResult(
            character=spec,
            person_model=person_model,
            cues=cues,
            retrieval=retrieval,
            retrieval_mode=self.retriever.effective_mode,
        )


def load_benchmark(path: str | Path) -> list[BenchmarkExample]:
    examples = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                examples.append(BenchmarkExample.model_validate_json(line))
            except Exception as exc:  # noqa: BLE001
                raise ValueError(f"invalid benchmark row at {path}:{line_no}: {exc}") from exc
    return examples


def _probe_question(probe_id: str, language: str) -> str:
    probe = PROBES_BY_ID[probe_id]
    return probe.question_zh if language.startswith("zh") else probe.question_en


def _probe_comments(items: list[dict]) -> list[dict[str, str]]:
    return [
        {
            "comment_id": str(item["comment_id"]),
            "platform": str(item.get("platform", "")),
            "observer": str(item.get("author_hash", "")),
            "text": str(item["text"]),
        }
        for item in items
    ]


def _mask_condition_output(text: str, *, condition: str, result: ProfileResult) -> str:
    text = anonymize_text(text, result.character)
    leaks = find_identity_leaks(text, result.character)
    if leaks:
        raise RuntimeError(f"{condition} payload leaked private identity strings: {leaks}")
    return text


def _condition_payload(
    llm: LLMClient | MockLLMClient,
    *,
    condition: str,
    result: ProfileResult,
) -> str:
    if condition not in SUPPORTED_CONDITIONS:
        raise ValueError(f"unknown condition: {condition}")
    if condition == "none":
        return ""
    if condition == "ours":
        return render_person_model(result.person_model)
    if condition == "gold":
        if not result.character.gold_profile:
            raise RuntimeError(f"gold condition requested but {result.character.character_id} has no gold profile")
        return anonymize_text(result.character.gold_profile, result.character)
    local_instruction = (
        "Summarize what this probe's third-party comments support about the anonymous person in "
        "relation to the supplied profiling question. Use only this probe's comments, preserve "
        "uncertainty and disagreement, do not guess identity, and do not invent facts or restate "
        "unrelated material."
        if condition == "summary"
        else "Extract only Big Five-relevant observations supported by this probe's third-party "
        "comments about the anonymous person. Discuss only supported dimensions among Openness, "
        "Conscientiousness, Extraversion, Agreeableness, and Neuroticism; preserve uncertainty and "
        "disagreement, do not guess identity, and do not add biography or unsupported traits."
    )
    aggregate_instruction = (
        "Integrate the supplied per-probe summaries into one anonymous generic person summary. Use "
        "only the supplied observations, preserve uncertainty, contradictions, and situational "
        "differences, do not guess identity, and do not invent facts."
        if condition == "summary"
        else "Integrate the supplied per-probe observations into one anonymous Big Five personality "
        "description covering Openness, Conscientiousness, Extraversion, Agreeableness, and "
        "Neuroticism. Use only supplied observations, preserve uncertainty and contradictions, do "
        "not guess identity, and do not add biography, motives, or unsupported traits."
    )
    observations = []
    language = str(result.character.language)
    for probe_id, items in result.retrieval.items():
        local_text = llm.chat(
            system=local_instruction,
            user=json.dumps(
                {
                    "anonymous_id": result.character.anonymous_id,
                    "probe_id": probe_id,
                    "profiling_question": _probe_question(probe_id, language),
                    "comments": _probe_comments(items),
                },
                ensure_ascii=False,
            ),
            agent=f"condition_{condition}:{probe_id}",
        )
        observations.append(
            {
                "probe_id": probe_id,
                "observation": _mask_condition_output(
                    local_text,
                    condition=condition,
                    result=result,
                ),
            }
        )
    text = llm.chat(
        system=aggregate_instruction,
        user=json.dumps(
            {
                "anonymous_id": result.character.anonymous_id,
                "probe_observations": observations,
            },
            ensure_ascii=False,
        ),
        agent=f"condition_{condition}:aggregate",
    )
    return _mask_condition_output(text, condition=condition, result=result)


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    payload["_config_path"] = str(config_path)
    return payload


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def _project_root(config_path: Path) -> Path:
    for candidate in config_path.resolve().parents:
        if (candidate / "pyproject.toml").is_file() and (candidate / "src" / "sideprofile").is_dir():
            return candidate
    raise RuntimeError(f"cannot locate project root above config: {config_path}")


class ExperimentRunner:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.config_path = Path(config["_config_path"])
        self.project_root = _project_root(self.config_path)
        self.run_dir: Path | None = None

    @classmethod
    def from_file(cls, path: str | Path) -> "ExperimentRunner":
        return cls(load_config(path))

    def run(self) -> Path:
        run_cfg = self.config.get("run", {})
        run_name = run_cfg.get("name", "experiment")
        data_cfg = self.config["data"]
        is_smoke = "smoke" in str(run_name).casefold()
        if not is_smoke:
            manifest_value = data_cfg.get("bundle_manifest")
            if not manifest_value:
                raise RuntimeError(
                    "research run requires data.bundle_manifest; freeze inputs locally first"
                )
            verification = verify_bundle_manifest(
                project_root=self.project_root,
                manifest_path=manifest_value,
            )
            if not verification["ok"]:
                raise RuntimeError(
                    "frozen bundle verification failed: "
                    + "; ".join(verification["failures"])
                )
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_id = f"{run_name}-{stamp}"
        output_root = _resolve(self.project_root, run_cfg.get("output_dir", "runs"))
        run_dir = output_root / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        self.run_dir = run_dir
        (run_dir / "status.json").write_text(
            json.dumps({"status": "running", "run_id": run_id}, indent=2),
            encoding="utf-8",
        )
        shutil.copy2(self.config_path, run_dir / "config.yaml")

        db_path = _resolve(self.project_root, data_cfg["corpus_db"])
        catalog_path = _resolve(self.project_root, data_cfg["catalog"])
        benchmark_path = _resolve(self.project_root, data_cfg["benchmark"])
        provider_cfg = self.config.get("provider", {})
        settings = ProviderSettings.from_env(
            provider_cfg.get("name", "GPT"),
            _resolve(self.project_root, provider_cfg.get("env_file", ".env")),
        )
        llm = LLMClient(settings, trace_path=run_dir / "trace.jsonl")
        retrieval_cfg = self.config.get("retrieval", {})
        vector_store_value = retrieval_cfg.get("vector_store")
        env_file = _resolve(self.project_root, provider_cfg.get("env_file", ".env"))
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
            rerank_trace_path=str(run_dir / "rerank.trace.jsonl"),
            candidate_top_k=int(retrieval_cfg.get("candidate_top_k", 20)),
            final_top_k=int(retrieval_cfg.get("final_top_k", 10)),
        )
        profiling_cfg = self.config.get("profiling", {})
        probes = select_probes(profiling_cfg.get("probe_ids"))
        include_synthetic = bool(data_cfg.get("include_synthetic", False))
        examples = load_benchmark(benchmark_path)
        character_ids = run_cfg.get("character_ids") or sorted(
            {example.character_id for example in examples}
        )
        conditions = run_cfg.get("conditions", ["ours"])
        unsupported = [condition for condition in conditions if condition not in SUPPORTED_CONDITIONS]
        if unsupported:
            raise ValueError(
                "unsupported conditions: "
                + ", ".join(unsupported)
                + ". Executable conditions are: "
                + ", ".join(SUPPORTED_CONDITIONS)
            )
        replicates = int(run_cfg.get("replicates", 3))
        if replicates < 1:
            raise ValueError("run.replicates must be at least 1")
        judge = bool(run_cfg.get("judge", True))
        catalog = {
            spec.character_id: spec for spec in load_character_catalog(catalog_path)
        }

        with CommentCorpus(db_path, read_only=True) as corpus:
            corpus.initialize()
            coverage_cfg = data_cfg.get("coverage")
            coverage_audit = None
            if coverage_cfg:
                coverage_audit = corpus.validate_targets(
                    min_comments=int(coverage_cfg["min_comments"]),
                    min_platforms=int(coverage_cfg["min_platforms"]),
                    min_authors=int(coverage_cfg["min_authors"]),
                    include_synthetic=include_synthetic,
                    character_ids=character_ids,
                )
                if not coverage_audit["ready"]:
                    failed = [
                        f"{row['character_id']} ({', '.join(row['failures'])})"
                        for row in coverage_audit["characters"]
                        if not row["ready"]
                    ]
                    raise RuntimeError("corpus coverage gate failed: " + "; ".join(failed))
            builder = ProfileBuilder(
                corpus,
                llm,
                retriever,
                characters=catalog,
                include_synthetic=include_synthetic,
            )
            manifest = {
                "run_id": run_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "model": settings.model,
                "provider": settings.provider,
                "characters": character_ids,
                "conditions": conditions,
                "replicates": replicates,
                "decoding": "provider_default",
                "probe_ids": [probe.probe_id for probe in probes],
                "include_synthetic": include_synthetic,
                "research_valid": not include_synthetic,
                "corpus_stats": corpus.stats(include_synthetic=include_synthetic),
                "coverage_audit": coverage_audit,
            }
            (run_dir / "manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            generation_path = run_dir / "generations.jsonl"
            for character_id in character_ids:
                result = builder.build(character_id, probes)
                artifact_dir = run_dir / "profiles" / character_id
                artifact_dir.mkdir(parents=True, exist_ok=True)
                (artifact_dir / "person_model.json").write_text(
                    result.person_model.model_dump_json(indent=2), encoding="utf-8"
                )
                with (artifact_dir / "cues.jsonl").open("w", encoding="utf-8") as handle:
                    for cue in result.cues:
                        handle.write(cue.model_dump_json() + "\n")
                (artifact_dir / "retrieval.json").write_text(
                    json.dumps(result.retrieval, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                (artifact_dir / "coverage.json").write_text(
                    json.dumps(cue_coverage(result.cues), ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                conditioning = {
                    condition: _condition_payload(llm, condition=condition, result=result)
                    for condition in conditions
                }
                for example in [item for item in examples if item.character_id == character_id]:
                    for condition in conditions:
                        for replicate in range(1, replicates + 1):
                            output = generate_response(
                                llm,
                                example=example,
                                model=result.person_model if condition == "ours" else None,
                                condition=condition,
                                conditioning_text=conditioning[condition],
                            )
                            output = anonymize_text(output, result.character)
                            score = rationale = None
                            if judge:
                                score, rationale = judge_response(
                                    llm,
                                    spec=result.character,
                                    example=example,
                                    output=output,
                                )
                            record = GenerationRecord(
                                run_id=run_id,
                                example_id=example.example_id,
                                character_id=character_id,
                                anonymous_id=result.character.anonymous_id,
                                condition=condition,
                                replicate=replicate,
                                query=example.query,
                                context=example.context,
                                output=output,
                                actor_model=settings.model,
                                score=score,
                                judge_rationale=rationale or "",
                            )
                            with generation_path.open("a", encoding="utf-8") as handle:
                                handle.write(record.model_dump_json() + "\n")

            manifest["llm_usage"] = llm.usage.__dict__
            manifest["retrieval_mode"] = retriever.effective_mode
            (run_dir / "manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        build_report(run_dir)
        output_root.mkdir(parents=True, exist_ok=True)
        (output_root / f"{run_name}-latest.json").write_text(
            json.dumps({"run_dir": str(run_dir)}, indent=2), encoding="utf-8"
        )
        (run_dir / "status.json").write_text(
            json.dumps({"status": "completed", "run_id": run_id}, indent=2),
            encoding="utf-8",
        )
        return run_dir
