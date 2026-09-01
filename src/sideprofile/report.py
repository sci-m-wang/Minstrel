from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_report(run_dir: str | Path) -> Path:
    run_path = Path(run_dir).resolve()
    manifest = json.loads((run_path / "manifest.json").read_text(encoding="utf-8"))
    records = _load_jsonl(run_path / "generations.jsonl")
    grouped: dict[str, list[float]] = defaultdict(list)
    counts: dict[str, int] = defaultdict(int)
    for record in records:
        condition = record["condition"]
        counts[condition] += 1
        if isinstance(record.get("score"), (int, float)):
            grouped[condition].append(float(record["score"]))
    rows = []
    for condition in sorted(counts):
        values = grouped[condition]
        rows.append(
            {
                "condition": condition,
                "responses": counts[condition],
                "scored": len(values),
                "mean_score": statistics.fmean(values) if values else None,
                "std_score": statistics.stdev(values) if len(values) > 1 else 0.0 if values else None,
            }
        )
    summary_path = run_path / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["condition", "responses", "scored", "mean_score", "std_score"]
        )
        writer.writeheader()
        writer.writerows(rows)

    research_valid = bool(manifest.get("research_valid", False))
    lines = [
        f"# SideProfile Run Report: {manifest['run_id']}",
        "",
        "## Validity status",
        "",
        (
            "This run excludes synthetic comments and is eligible for research analysis, subject to corpus licensing and benchmark checks."
            if research_valid
            else "**SMOKE TEST ONLY.** This run includes synthetic comments and must not be reported as experimental evidence."
        ),
        "",
        "## Configuration",
        "",
        f"- Provider/model: `{manifest.get('provider')}` / `{manifest.get('model')}`",
        f"- Generation settings: `{manifest.get('decoding', 'provider_default')}`",
        f"- Retrieval: `{manifest.get('retrieval_mode', 'unknown')}`",
        f"- Characters: {', '.join(manifest.get('characters', []))}",
        f"- Conditions: {', '.join(manifest.get('conditions', []))}",
        f"- Independent replicates: {manifest.get('replicates', 1)}",
        f"- Profiling probes: {len(manifest.get('probe_ids', []))}",
        "",
        "## Aggregate results",
        "",
        "| Condition | Responses | Scored | Mean | SD |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        mean = "—" if row["mean_score"] is None else f"{row['mean_score']:.4f}"
        std = "—" if row["std_score"] is None else f"{row['std_score']:.4f}"
        lines.append(
            f"| {row['condition']} | {row['responses']} | {row['scored']} | {mean} | {std} |"
        )
    lines.extend(
        [
            "",
            "## Corpus snapshot",
            "",
            f"- Comments visible to this run: {manifest.get('corpus_stats', {}).get('total_comments', 0)}",
            f"- Synthetic comments excluded by default: {manifest.get('corpus_stats', {}).get('synthetic_comments_excluded', 0)}",
            "",
            "## Interpretation guardrails",
            "",
            "- Treat model-judge scores as estimates; compare with the official benchmark evaluator before publication.",
            "- Report corpus coverage, platforms, independent authors, retrieval mode, model versions, replicates, and failed calls.",
            "- Treat Summary, Personality, Gold, and Ours as distinct internal conditions with their native payloads; do not claim that their information or lengths are normalized.",
            "- Inspect identity leakage separately from role fidelity.",
            "",
            "## Artifacts",
            "",
            "- `manifest.json`: frozen run metadata and token/call usage",
            f"- Prepared conditionings: `{manifest.get('prepared_dir', 'see prepared_manifest_sha256')}`",
            f"- Prepared manifest SHA-256: `{manifest.get('prepared_manifest_sha256', 'unknown')}`",
            "- The prepared directory contains identity-blind person models, evidence-cited cues, retrieval records, and all five immutable conditionings.",
            "- `generations.jsonl`: actor outputs and judge records",
            "- `summary.csv`: machine-readable aggregate table",
        ]
    )
    report_path = run_path / "report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path
