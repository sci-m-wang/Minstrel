#!/usr/bin/env python3
"""Score one Panel D run with the official Baichuan CharacterRM procedure.

The prompt layout, left truncation to the official 4096-token context, and ``score * 4 + 1``
transformation follow CharacterEval's published ``run_char_rm.py``. Model and tokenizer loading is
strictly local so a disconnected compute node never attempts a download.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


OFFICIAL_CONTEXT_LENGTH = 4096
PRIMARY_METRICS = {"Behavior", "Utterance", "Humanlikeness", "Consistency"}


def format_input(record: dict, profiles: dict) -> str:
    return (
        "<RoleInfo>\n\n"
        + str(profiles[record["role"]])
        + "\n\n<Context>\n\n"
        + record["context"]
        + "\n\n<Response>\n\n"
        + record["model_output"]
        + "\n\n<Dimension>\n\n"
        + record["metric_zh"]
    )


def aggregate(records: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, int, str, str], list[float]] = defaultdict(list)
    for record in records:
        grouped[
            (
                str(record["condition"]),
                int(record["replicate"]),
                str(record["actor_model"]),
                str(record["metric_en"]),
            )
        ].append(float(record["character_rm_score"]))
    return [
        {
            "condition": condition,
            "replicate": replicate,
            "actor_model": actor_model,
            "metric": metric,
            "primary_metric": metric in PRIMARY_METRICS,
            "examples": len(values),
            "mean": statistics.fmean(values),
            "standard_deviation": statistics.stdev(values) if len(values) > 1 else 0.0,
        }
        for (condition, replicate, actor_model, metric), values in sorted(grouped.items())
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--rm-path", required=True)
    parser.add_argument(
        "--profiles",
        default="data/evaluators/charactereval/character_profiles.selected.json",
    )
    args = parser.parse_args()

    import torch  # noqa: PLC0415
    import tqdm  # noqa: PLC0415
    from transformers import AutoModel, AutoTokenizer  # noqa: PLC0415

    run_dir = Path(args.run_dir).resolve()
    rm_path = Path(args.rm_path).resolve()
    input_path = run_dir / "charactereval-rm-input.json"
    profiles = json.loads(Path(args.profiles).read_text(encoding="utf-8"))
    records = json.loads(input_path.read_text(encoding="utf-8"))

    tokenizer = AutoTokenizer.from_pretrained(
        rm_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModel.from_pretrained(
        rm_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        local_files_only=True,
    ).cuda().eval()

    scored: list[dict] = []
    for record in tqdm.tqdm(records, desc="CharacterRM"):
        input_ids = tokenizer.encode(
            text=format_input(record, profiles), add_special_tokens=False
        ) + [tokenizer.eos_token_id]
        input_ids = input_ids[-OFFICIAL_CONTEXT_LENGTH:]
        tensor = torch.tensor(input_ids).unsqueeze(0).cuda()
        with torch.no_grad():
            score = model(input_ids=tensor)[1].item() * 4 + 1
        item = dict(record)
        item[record["metric_en"]] = score
        item["character_rm_score"] = score
        scored.append(item)

    payload = {
        "benchmark": "CharacterEval",
        "evaluator": "morecry/BaichuanCharRM official procedure",
        "official_context_length": OFFICIAL_CONTEXT_LENGTH,
        "primary_metrics": sorted(PRIMARY_METRICS),
        "summary": aggregate(scored),
        "records": scored,
    }
    output = run_dir / "charactereval-evaluation.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"status": "ok", "output": str(output), "records": len(scored)},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
