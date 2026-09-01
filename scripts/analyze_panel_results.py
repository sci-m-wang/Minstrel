#!/usr/bin/env python3
"""Aggregate completed actor runs using only the panel's official evaluator artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import yaml


CONTRAST_BASELINES = ("summary", "personality", "gold")
EXPECTED_CONDITIONS = ("none", "personality", "summary", "gold", "ours")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def interval(values: list[float]) -> tuple[float, float, float, float]:
    mean = statistics.fmean(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    half_width = 1.96 * sd / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return mean, sd, mean - half_width, mean + half_width


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def collect_panel_a(run_dir: Path, actor: str) -> tuple[list[dict], dict]:
    evaluation = load_json(run_dir / "roleagentbench-evaluation.json")
    rows = [
        {
            "actor_model": actor,
            "metric": "ReactionAccuracy",
            "condition": item["condition"],
            "replicate": int(item["replicate"]),
            "example_id": item["example_id"],
            "character_id": item["character_id"],
            "value": float(bool(item["correct"])),
        }
        for item in evaluation["records"]
    ]
    return rows, {
        "reaction_evaluator": evaluation.get("reaction_evaluator"),
        "general_response_evaluator": evaluation.get("general_response_evaluator"),
        "general_response_generations": evaluation.get("general_response_generations", 0),
    }


def collect_panel_d(run_dir: Path, actor: str) -> tuple[list[dict], dict]:
    evaluation = load_json(run_dir / "charactereval-evaluation.json")
    rows = [
        {
            "actor_model": actor,
            "metric": item["metric_en"],
            "condition": item["condition"],
            "replicate": int(item["replicate"]),
            "example_id": item["example_id"],
            "character_id": item["character_id"],
            "value": float(item["character_rm_score"]),
        }
        for item in evaluation["records"]
    ]
    return rows, {
        "evaluator": evaluation.get("evaluator"),
        "official_context_length": evaluation.get("official_context_length"),
        "primary_metrics": evaluation.get("primary_metrics", []),
    }


def summarize(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(row["actor_model"], row["metric"], row["condition"])].append(row["value"])
        grouped[("ALL", row["metric"], row["condition"])].append(row["value"])
    output = []
    for (actor, metric, condition), values in sorted(grouped.items()):
        mean, sd, low, high = interval(values)
        output.append(
            {
                "actor_model": actor,
                "metric": metric,
                "condition": condition,
                "aggregation": "official_micro",
                "n": len(values),
                "mean": mean,
                "standard_deviation": sd,
                "ci95_low": low,
                "ci95_high": high,
            }
        )
    macro_cells: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        macro_cells[
            (row["actor_model"], row["metric"], row["condition"], row["character_id"])
        ].append(row["value"])
    macro_grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for (actor, metric, condition, _character), values in macro_cells.items():
        macro_grouped[(actor, metric, condition)].append(statistics.fmean(values))
        macro_grouped[("ALL", metric, condition)].append(statistics.fmean(values))
    for (actor, metric, condition), values in sorted(macro_grouped.items()):
        mean, sd, low, high = interval(values)
        output.append(
            {
                "actor_model": actor,
                "metric": metric,
                "condition": condition,
                "aggregation": "character_macro",
                "n": len(values),
                "mean": mean,
                "standard_deviation": sd,
                "ci95_low": low,
                "ci95_high": high,
            }
        )
    return output


def paired_contrasts(rows: list[dict]) -> tuple[list[dict], list[str]]:
    indexed: dict[tuple[str, str, int, str, str], float] = {}
    character_by_key: dict[tuple[str, str, int, str, str], str] = {}
    duplicates: list[str] = []
    for row in rows:
        key = (
            row["actor_model"],
            row["example_id"],
            row["replicate"],
            row["metric"],
            row["condition"],
        )
        if key in indexed:
            duplicates.append("|".join(map(str, key)))
        indexed[key] = row["value"]
        character_by_key[key] = row["character_id"]

    actors = sorted({row["actor_model"] for row in rows})
    metrics = sorted({row["metric"] for row in rows})
    output = []
    missing_pairs: list[str] = []
    for actor_scope in [*actors, "ALL"]:
        scoped_actors = actors if actor_scope == "ALL" else [actor_scope]
        for metric in metrics:
            for baseline in CONTRAST_BASELINES:
                differences = []
                for key, ours in indexed.items():
                    actor, example_id, replicate, row_metric, condition = key
                    if actor not in scoped_actors or row_metric != metric or condition != "ours":
                        continue
                    pair_key = (actor, example_id, replicate, metric, baseline)
                    if pair_key not in indexed:
                        missing_pairs.append("|".join(map(str, pair_key)))
                        continue
                    differences.append(ours - indexed[pair_key])
                if not differences:
                    continue
                mean, sd, low, high = interval(differences)
                output.append(
                    {
                        "actor_model": actor_scope,
                        "metric": metric,
                        "contrast": f"ours-{baseline}",
                        "aggregation": "exact_unit_micro",
                        "paired_n": len(differences),
                        "mean_difference": mean,
                        "standard_deviation": sd,
                        "ci95_low": low,
                        "ci95_high": high,
                        "ci_method": "paired normal-approximation",
                    }
                )
                macro_cells: dict[tuple[str, str], list[float]] = defaultdict(list)
                for key, ours in indexed.items():
                    actor, example_id, replicate, row_metric, condition = key
                    if actor not in scoped_actors or row_metric != metric or condition != "ours":
                        continue
                    pair_key = (actor, example_id, replicate, metric, baseline)
                    if pair_key not in indexed:
                        continue
                    character = character_by_key[key]
                    macro_cells[(actor, character)].append(ours - indexed[pair_key])
                macro_differences = [statistics.fmean(values) for values in macro_cells.values()]
                if macro_differences:
                    mean, sd, low, high = interval(macro_differences)
                    output.append(
                        {
                            "actor_model": actor_scope,
                            "metric": metric,
                            "contrast": f"ours-{baseline}",
                            "aggregation": "character_macro",
                            "paired_n": len(macro_differences),
                            "mean_difference": mean,
                            "standard_deviation": sd,
                            "ci95_low": low,
                            "ci95_high": high,
                            "ci_method": "paired normal-approximation",
                        }
                    )
    return output, sorted(set([*duplicates, *missing_pairs]))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", choices=["A", "D"], required=True)
    parser.add_argument("--run-dir", action="append", required=True)
    parser.add_argument("--model-registry", default="offline/models.yaml")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    registry = yaml.safe_load(Path(args.model_registry).read_text(encoding="utf-8"))
    panel_key = f"panel_{args.panel.lower()}"
    expected_actors = list(registry["actor_matrix"][panel_key])
    rows: list[dict] = []
    run_records = []
    failures: list[str] = []
    evaluator_details = []
    observed_actors: list[str] = []

    for value in args.run_dir:
        run_dir = Path(value).resolve()
        status = load_json(run_dir / "status.json")
        manifest = load_json(run_dir / "manifest.json")
        generations = load_jsonl(run_dir / "generations.jsonl")
        actor = str(manifest.get("model", ""))
        observed_actors.append(actor)
        generation_keys = {
            (
                item["example_id"],
                item["character_id"],
                item["condition"],
                int(item["replicate"]),
            )
            for item in generations
        }
        if status.get("status") != "completed":
            failures.append(f"{run_dir}: status is not completed")
        if manifest.get("research_valid") is not True:
            failures.append(f"{run_dir}: research_valid is not true")
        if tuple(manifest.get("conditions", [])) != EXPECTED_CONDITIONS:
            failures.append(f"{run_dir}: condition order/scope differs")
        expected_generation_count = (
            len({item["example_id"] for item in generations})
            * len(EXPECTED_CONDITIONS)
            * int(manifest.get("replicates", 0))
        )
        if not generations:
            failures.append(f"{run_dir}: no generations")
        if len(generation_keys) != len(generations):
            failures.append(f"{run_dir}: duplicate generation cells")
        if len(generations) != expected_generation_count:
            failures.append(
                f"{run_dir}: generations={len(generations)} expected={expected_generation_count}"
            )
        try:
            if args.panel == "A":
                collected, details = collect_panel_a(run_dir, actor)
            else:
                collected, details = collect_panel_d(run_dir, actor)
        except (FileNotFoundError, KeyError, ValueError) as exc:
            failures.append(f"{run_dir}: official evaluator artifact invalid or missing: {exc}")
            collected, details = [], {}
        rows.extend(collected)
        evaluator_details.append(details)
        run_records.append(
            {
                "run_dir": str(run_dir),
                "run_id": manifest.get("run_id"),
                "actor_model": actor,
                "prepared_manifest_sha256": manifest.get("prepared_manifest_sha256"),
                "generation_records": len(generations),
                "official_score_records": len(collected),
            }
        )

    missing_actors = [item for item in expected_actors if item not in observed_actors]
    unexpected_actors = [item for item in observed_actors if item not in expected_actors]
    duplicate_actors = sorted({item for item in observed_actors if observed_actors.count(item) > 1})
    if missing_actors:
        failures.append("missing pre-registered actors: " + ", ".join(missing_actors))
    if unexpected_actors:
        failures.append("unexpected actors: " + ", ".join(unexpected_actors))
    if duplicate_actors:
        failures.append("duplicate actor runs: " + ", ".join(duplicate_actors))
    prepared_hashes = {item["prepared_manifest_sha256"] for item in run_records}
    if len(prepared_hashes) != 1:
        failures.append("actor runs do not share one prepared-conditioning hash")

    summary_rows = summarize(rows)
    contrast_rows, pairing_failures = paired_contrasts(rows)
    if pairing_failures:
        failures.append(f"missing or duplicate paired cells: {len(pairing_failures)}")

    summary_fields = [
        "actor_model", "metric", "condition", "aggregation", "n", "mean", "standard_deviation",
        "ci95_low", "ci95_high",
    ]
    contrast_fields = [
        "actor_model", "metric", "contrast", "aggregation", "paired_n", "mean_difference",
        "standard_deviation", "ci95_low", "ci95_high", "ci_method",
    ]
    write_csv(output_dir / "condition-summary.csv", summary_rows, summary_fields)
    write_csv(output_dir / "paired-contrasts.csv", contrast_rows, contrast_fields)

    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "panel": args.panel,
        "research_valid": not failures,
        "expected_actors": expected_actors,
        "observed_actors": observed_actors,
        "failures": failures,
        "pairing_failure_examples": pairing_failures[:20],
        "runs": run_records,
        "evaluator_details": evaluator_details,
        "condition_summary": summary_rows,
        "paired_contrasts": contrast_rows,
    }
    (output_dir / "analysis.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        f"# SideProfile Panel {args.panel} Official Results",
        "",
        f"- Research-valid aggregate: **{str(not failures).lower()}**",
        f"- Expected actors: {', '.join(expected_actors)}",
        f"- Observed actors: {', '.join(observed_actors) or 'none'}",
        f"- Shared prepared-conditioning SHA-256: {next(iter(prepared_hashes), 'missing')}",
        "- Confidence intervals: paired or unpaired normal approximation, computed without random resampling or a seed.",
    ]
    if args.panel == "A":
        lines.append("- General Response official human/GPT-4 pairwise evaluation remains pending offline.")
    if failures:
        lines.extend(["", "## Validity failures", "", *[f"- {item}" for item in failures]])
    lines.extend(
        [
            "",
            "## Condition summary",
            "",
            "See `condition-summary.csv` for condition means, standard deviations, counts, and 95% intervals.",
            "",
            "## Pre-registered paired contrasts",
            "",
            "See `paired-contrasts.csv` for Ours minus Summary, Personality, and Gold on exact matched units.",
            "",
            "## Run inventory",
            "",
            *[
                f"- `{item['actor_model']}`: `{item['run_dir']}` ({item['generation_records']} generations)"
                for item in run_records
            ],
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"status": "ok" if not failures else "incomplete", "output_dir": str(output_dir), "failures": failures},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
