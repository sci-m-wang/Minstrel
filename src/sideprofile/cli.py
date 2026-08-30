from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .bundle import build_bundle_manifest, verify_bundle_manifest
from .corpus import CommentCorpus, load_character_catalog, load_comments
from .llm import LLMClient, ProviderSettings
from .pipeline import ExperimentRunner, ProfileBuilder
from .probes import select_probes
from .report import build_report
from .retrieval import HybridRetriever
from .staged import ActorRunner, ConditioningPreparer, verify_prepared
from .vector_store import build_vector_store, verify_vector_store


def _json(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2))


def init_corpus(args: argparse.Namespace) -> int:
    with CommentCorpus(args.db) as corpus:
        corpus.initialize()
        if args.catalog:
            specs = load_character_catalog(args.catalog)
            corpus.add_characters(specs)
        _json({"status": "ok", "db": str(Path(args.db).resolve()), "characters": len(corpus.list_characters())})
    return 0


def import_comments(args: argparse.Namespace) -> int:
    comments = load_comments(args.input)
    if any(comment.is_synthetic for comment in comments) and not args.allow_synthetic:
        raise RuntimeError("input contains synthetic comments; pass --allow-synthetic only for smoke tests")
    with CommentCorpus(args.db) as corpus:
        corpus.initialize()
        if args.catalog:
            corpus.add_characters(load_character_catalog(args.catalog))
        inserted, duplicate = corpus.add_comments(comments)
        _json({"status": "ok", "inserted": inserted, "duplicates": duplicate})
    return 0


def corpus_stats(args: argparse.Namespace) -> int:
    with CommentCorpus(args.db) as corpus:
        corpus.initialize()
        _json(corpus.stats(include_synthetic=args.include_synthetic))
    return 0


def validate_corpus(args: argparse.Namespace) -> int:
    with CommentCorpus(args.db) as corpus:
        corpus.initialize()
        if args.catalog:
            corpus.add_characters(load_character_catalog(args.catalog))
        result = corpus.validate_targets(
            min_comments=args.min_comments,
            min_platforms=args.min_platforms,
            min_authors=args.min_authors,
            include_synthetic=args.include_synthetic,
            character_ids=(
                [item.strip() for item in args.character_ids.split(",") if item.strip()]
                if args.character_ids
                else None
            ),
        )
        _json(result)
    return 0 if result["ready"] else 2


def doctor(args: argparse.Namespace) -> int:
    settings = ProviderSettings.from_env(args.provider, args.env_file)
    result = {
        "status": "configured",
        "provider": settings.provider,
        "model": settings.model,
        "base_url_configured": bool(settings.base_url),
        "api_key_present": bool(settings.api_key),
    }
    if args.live:
        client = LLMClient(settings)
        result["live_test"] = client.probe()
    _json(result)
    return 0


def build_profile_command(args: argparse.Namespace) -> int:
    settings = ProviderSettings.from_env(args.provider, args.env_file)
    llm = LLMClient(settings, trace_path=args.trace)
    retriever = HybridRetriever(
        mode=args.retrieval,
        vector_store=args.vector_store,
        embedding_model_key=args.embedding_model_key,
        reranker_model=args.reranker_model,
        final_top_k=args.top_k,
    )
    probes = select_probes(args.probes.split(",") if args.probes else None)
    with CommentCorpus(args.db) as corpus:
        builder = ProfileBuilder(
            corpus,
            llm,
            retriever,
            include_synthetic=args.include_synthetic,
            target_tokens=args.target_tokens,
        )
        result = builder.build(args.character_id, probes)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(result.person_model.model_dump_json(indent=2), encoding="utf-8")
    cue_path = output.with_suffix(".cues.jsonl")
    with cue_path.open("w", encoding="utf-8") as handle:
        for cue in result.cues:
            handle.write(cue.model_dump_json() + "\n")
    _json(
        {
            "status": "ok",
            "person_model": str(output.resolve()),
            "cues": str(cue_path.resolve()),
            "retrieval_mode": result.retrieval_mode,
            "llm_usage": llm.usage.__dict__,
        }
    )
    return 0


def build_vector_store_command(args: argparse.Namespace) -> int:
    result = build_vector_store(
        corpus_db=args.db,
        output_path=args.output,
        model_path=args.model,
        model_key=args.model_key,
        model_revision=args.model_revision,
    )
    _json(result)
    return 0


def verify_vector_store_command(args: argparse.Namespace) -> int:
    result = verify_vector_store(
        vector_store=args.vector_store,
        corpus_db=args.db,
        expected_model_key=args.model_key,
        expected_model_revision=args.model_revision,
    )
    _json(result)
    return 0 if result["ok"] else 2


def run_experiment(args: argparse.Namespace) -> int:
    runner = ExperimentRunner.from_file(args.config)
    try:
        run_dir = runner.run()
    except BaseException as exc:
        if runner.run_dir:
            (runner.run_dir / "status.json").write_text(
                json.dumps(
                    {
                        "status": "failed",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        raise
    _json({"status": "ok", "run_dir": str(run_dir), "report": str(run_dir / "report.md")})
    return 0


def prepare_conditionings(args: argparse.Namespace) -> int:
    runner = ConditioningPreparer.from_file(args.config)
    try:
        output = runner.run()
    except BaseException as exc:
        if runner.output_dir:
            (runner.output_dir / "status.json").write_text(
                json.dumps(
                    {
                        "status": "failed",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        raise
    _json({"status": "ok", "prepared_dir": str(output)})
    return 0


def verify_conditionings(args: argparse.Namespace) -> int:
    result = verify_prepared(args.prepared_dir)
    _json(result)
    return 0 if result["ok"] else 2


def run_actor(args: argparse.Namespace) -> int:
    runner = ActorRunner.from_file(args.config, args.prepared_dir)
    try:
        run_dir = runner.run()
    except BaseException as exc:
        if runner.run_dir:
            (runner.run_dir / "status.json").write_text(
                json.dumps(
                    {
                        "status": "failed",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        raise
    _json({"status": "ok", "run_dir": str(run_dir), "report": str(run_dir / "report.md")})
    return 0


def analyze(args: argparse.Namespace) -> int:
    report = build_report(args.run_dir)
    _json({"status": "ok", "report": str(report)})
    return 0


def freeze_bundle(args: argparse.Namespace) -> int:
    manifest = build_bundle_manifest(
        project_root=args.project_root,
        config_path=args.config,
        output_path=args.output,
    )
    _json(
        {
            "status": "ok",
            "output": str(Path(args.output).resolve()),
            "run_name": manifest["run_name"],
            "research_ready": manifest["research_ready"],
        }
    )
    return 0


def verify_bundle(args: argparse.Namespace) -> int:
    result = verify_bundle_manifest(
        project_root=args.project_root,
        manifest_path=args.manifest,
    )
    _json(result)
    return 0 if result["ok"] else 2


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sideprofile")
    sub = parser.add_subparsers(dest="command", required=True)

    command = sub.add_parser("init-corpus", help="initialize SQLite corpus and optional character catalog")
    command.add_argument("--db", required=True)
    command.add_argument("--catalog")
    command.set_defaults(func=init_corpus)

    command = sub.add_parser("import-comments", help="validate and import JSONL/CSV comments")
    command.add_argument("--db", required=True)
    command.add_argument("--input", required=True)
    command.add_argument("--catalog")
    command.add_argument("--allow-synthetic", action="store_true")
    command.set_defaults(func=import_comments)

    command = sub.add_parser("corpus-stats", help="show auditable corpus counts")
    command.add_argument("--db", required=True)
    command.add_argument("--include-synthetic", action="store_true")
    command.set_defaults(func=corpus_stats)

    command = sub.add_parser("validate-corpus", help="check coverage targets before a research run")
    command.add_argument("--db", required=True)
    command.add_argument("--catalog")
    command.add_argument("--min-comments", type=int, default=500)
    command.add_argument("--min-platforms", type=int, default=2)
    command.add_argument("--min-authors", type=int, default=100)
    command.add_argument(
        "--character-ids",
        help="comma-separated configured character IDs; default validates every character in the DB",
    )
    command.add_argument("--include-synthetic", action="store_true")
    command.set_defaults(func=validate_corpus)

    command = sub.add_parser("doctor", help="validate .env and optionally make one live GPT call")
    command.add_argument("--provider", default="GPT")
    command.add_argument("--env-file", default=".env")
    command.add_argument("--live", action="store_true")
    command.set_defaults(func=doctor)

    command = sub.add_parser("build-profile", help="build one identity-blind person model")
    command.add_argument("--db", required=True)
    command.add_argument("--character-id", required=True)
    command.add_argument("--output", required=True)
    command.add_argument("--provider", default="GPT")
    command.add_argument("--env-file", default=".env")
    command.add_argument("--retrieval", choices=["auto", "bm25", "hybrid"], default="auto")
    command.add_argument("--vector-store")
    command.add_argument("--embedding-model-key", default="qwen3-embedding-0.6b")
    command.add_argument("--reranker-model")
    command.add_argument("--top-k", type=int, default=10)
    command.add_argument("--probes", help="comma-separated probe IDs; default is all 24")
    command.add_argument("--target-tokens", type=int, default=1000)
    command.add_argument("--trace")
    command.add_argument("--include-synthetic", action="store_true")
    command.set_defaults(func=build_profile_command)

    command = sub.add_parser(
        "build-vector-store",
        help="precompute the immutable exact Qwen3 comment/query vector database",
    )
    command.add_argument("--db", required=True)
    command.add_argument("--output", required=True)
    command.add_argument("--model", required=True)
    command.add_argument("--model-key", required=True)
    command.add_argument("--model-revision", required=True)
    command.set_defaults(func=build_vector_store_command)

    command = sub.add_parser(
        "verify-vector-store",
        help="verify a frozen vector database against the current research corpus",
    )
    command.add_argument("--db", required=True)
    command.add_argument("--vector-store", required=True)
    command.add_argument("--model-key")
    command.add_argument("--model-revision")
    command.set_defaults(func=verify_vector_store_command)

    command = sub.add_parser("run", help="execute configured profiling, generation, judging, and reporting")
    command.add_argument("--config", required=True)
    command.set_defaults(func=run_experiment)

    command = sub.add_parser(
        "prepare-conditionings",
        help="build and checksum the shared six-condition treatment with the fixed profiler",
    )
    command.add_argument("--config", required=True)
    command.set_defaults(func=prepare_conditionings)

    command = sub.add_parser(
        "verify-conditionings", help="verify shared conditionings before an actor run"
    )
    command.add_argument("--prepared-dir", required=True)
    command.set_defaults(func=verify_conditionings)

    command = sub.add_parser(
        "run-actor", help="run one actor over immutable prepared conditionings"
    )
    command.add_argument("--config", required=True)
    command.add_argument("--prepared-dir", required=True)
    command.set_defaults(func=run_actor)

    command = sub.add_parser("analyze", help="regenerate aggregate report from a completed run")
    command.add_argument("--run-dir", required=True)
    command.set_defaults(func=analyze)

    command = sub.add_parser(
        "freeze-bundle", help="freeze and checksum a completed research input bundle"
    )
    command.add_argument("--project-root", default=".")
    command.add_argument("--config", required=True)
    command.add_argument("--output", required=True)
    command.set_defaults(func=freeze_bundle)

    command = sub.add_parser(
        "verify-bundle", help="verify a frozen research input bundle without modifying it"
    )
    command.add_argument("--project-root", default=".")
    command.add_argument("--manifest", required=True)
    command.set_defaults(func=verify_bundle)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
