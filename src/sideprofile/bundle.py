from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .corpus import CommentCorpus, load_character_catalog
from .schema import BenchmarkExample


BUNDLE_SCHEMA_VERSION = 1


def _load_benchmark(path: Path) -> list[BenchmarkExample]:
    examples: list[BenchmarkExample] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                examples.append(BenchmarkExample.model_validate_json(line))
            except Exception as exc:  # noqa: BLE001
                raise ValueError(f"invalid benchmark row at {path}:{line_no}: {exc}") from exc
    return examples


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()


def _relative(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact(root: Path, path: Path, kind: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing {kind}: {path}")
    return {
        "kind": kind,
        "path": _relative(root, path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def build_bundle_manifest(
    *, project_root: str | Path, config_path: str | Path, output_path: str | Path
) -> dict[str, Any]:
    """Freeze and checksum all data inputs needed by one research configuration."""
    root = Path(project_root).resolve()
    config_file = _resolve(root, str(config_path))
    config = yaml.safe_load(config_file.read_text(encoding="utf-8")) or {}
    data = config.get("data", {})
    run = config.get("run", {})
    run_name = str(run.get("name", "experiment"))
    if "smoke" in run_name.casefold() or bool(data.get("include_synthetic", False)):
        raise ValueError("freeze-bundle is only for non-synthetic research configurations")

    db_path = _resolve(root, data["corpus_db"])
    catalog_path = _resolve(root, data["catalog"])
    benchmark_path = _resolve(root, data["benchmark"])
    scope_path = root / "configs" / "scope.yaml"
    output_file = _resolve(root, str(output_path))
    selected = list(run.get("character_ids") or [])
    if not selected:
        raise ValueError("run.character_ids must be explicit before freezing")

    catalog = {item.character_id: item for item in load_character_catalog(catalog_path)}
    unknown = sorted(set(selected) - set(catalog))
    if unknown:
        raise ValueError(f"catalog is missing selected characters: {', '.join(unknown)}")
    if "gold" in run.get("conditions", []):
        missing_gold = [cid for cid in selected if not catalog[cid].gold_profile]
        if missing_gold:
            raise ValueError(
                "gold condition is selected but gold_profile is missing for: "
                + ", ".join(missing_gold)
            )

    examples = _load_benchmark(benchmark_path)
    benchmark_counts = {
        cid: sum(example.character_id == cid for example in examples) for cid in selected
    }
    missing_benchmark = [cid for cid, count in benchmark_counts.items() if count == 0]
    if missing_benchmark:
        raise ValueError(
            "benchmark has no examples for selected characters: "
            + ", ".join(missing_benchmark)
        )

    coverage = data.get("coverage") or {}
    required_coverage = {
        "min_comments": int(coverage["min_comments"]),
        "min_platforms": int(coverage["min_platforms"]),
        "min_authors": int(coverage["min_authors"]),
    }
    with CommentCorpus(db_path) as corpus:
        audit = corpus.validate_targets(
            **required_coverage,
            include_synthetic=False,
            character_ids=selected,
        )
        if not audit["ready"]:
            failures = [
                f"{row['character_id']} ({', '.join(row['failures'])})"
                for row in audit["characters"]
                if not row["ready"]
            ]
            raise ValueError("corpus is not ready to freeze: " + "; ".join(failures))
        corpus.connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        source_rows = corpus.connection.execute(
            """
            SELECT character_id, platform, license_note,
                   COUNT(*) AS comments, COUNT(DISTINCT author_hash) AS authors
            FROM comments
            WHERE is_synthetic = 0
              AND character_id IN ({})
            GROUP BY character_id, platform, license_note
            ORDER BY character_id, platform, license_note
            """.format(",".join("?" for _ in selected)),
            selected,
        ).fetchall()
        source_audit = [dict(row) for row in source_rows]

    artifacts = [
        _artifact(root, config_file, "config"),
        _artifact(root, db_path, "corpus_db"),
        _artifact(root, catalog_path, "catalog"),
        _artifact(root, benchmark_path, "benchmark"),
        _artifact(root, scope_path, "executable_scope"),
    ]
    for value in data.get("frozen_assets", []):
        artifacts.append(_artifact(root, _resolve(root, value), "frozen_asset"))
    manifest = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_name": run_name,
        "research_ready": True,
        "read_only_execution": True,
        "character_ids": selected,
        "conditions": list(run.get("conditions", [])),
        "coverage_requirement": required_coverage,
        "coverage_audit": audit,
        "benchmark_examples": benchmark_counts,
        "source_audit": source_audit,
        "artifacts": artifacts,
    }
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def verify_bundle_manifest(
    *, project_root: str | Path, manifest_path: str | Path
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    manifest_file = _resolve(root, str(manifest_path))
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    failures: list[str] = []
    if manifest.get("schema_version") != BUNDLE_SCHEMA_VERSION:
        failures.append("unsupported manifest schema_version")
    if manifest.get("research_ready") is not True:
        failures.append("manifest is not marked research_ready")
    if manifest.get("read_only_execution") is not True:
        failures.append("manifest is not marked read_only_execution")
    for item in manifest.get("artifacts", []):
        path = _resolve(root, item["path"])
        if not path.is_file():
            failures.append(f"missing {item['kind']}: {item['path']}")
            continue
        observed = sha256_file(path)
        if observed != item["sha256"]:
            failures.append(f"checksum mismatch for {item['path']}")
    return {
        "ok": not failures,
        "manifest": _relative(root, manifest_file),
        "run_name": manifest.get("run_name", ""),
        "failures": failures,
    }
