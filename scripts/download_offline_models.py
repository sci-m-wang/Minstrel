#!/usr/bin/env python3
"""Download runtime-required pinned models into a relocatable offline asset directory.

No retry policy, worker override, quantization, or model-generation setting is introduced. The
declared repository client uses its defaults. Run this only on a connected preparation machine,
never on the GPU execution host.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import yaml


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_command(item: dict, destination: Path) -> list[str]:
    source = item.get("source", "huggingface")
    if source == "huggingface":
        command = [
            "hf",
            "download",
            item["repo_id"],
            "--revision",
            item["revision"],
            "--local-dir",
            str(destination),
        ]
        for pattern in item.get("include", []):
            command.extend(["--include", pattern])
        for pattern in item.get("exclude", []):
            command.extend(["--exclude", pattern])
        return command
    if source == "modelscope":
        command = [
            "modelscope",
            "download",
            "--model",
            item["repo_id"],
            "--revision",
            item.get("source_revision", item["revision"]),
            "--local_dir",
            str(destination),
        ]
        if item.get("include"):
            command.extend(["--include", *item["include"]])
        if item.get("exclude"):
            command.extend(["--exclude", *item["exclude"]])
        return command
    raise ValueError(f"unsupported model source: {source}")


def verify_modelscope_revision(item: dict) -> None:
    source_revision = item.get("source_revision")
    if not source_revision:
        raise ValueError("ModelScope entries require source_revision plus an exact revision commit")
    repository = f"https://www.modelscope.cn/{item['repo_id']}.git"
    result = subprocess.run(
        ["git", "ls-remote", repository, f"refs/heads/{source_revision}"],
        check=True,
        capture_output=True,
        text=True,
    )
    resolved = result.stdout.strip().partition("\t")[0]
    if resolved != item["revision"]:
        raise RuntimeError(
            f"ModelScope {item['repo_id']}@{source_revision} resolved to {resolved or 'nothing'}, "
            f"expected {item['revision']}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", default="offline/models.yaml")
    parser.add_argument("--asset-root", required=True)
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument(
        "--refresh-manifest",
        action="store_true",
        help="Refresh the spec hash/order from an existing complete manifest without downloading.",
    )
    args = parser.parse_args()

    spec = yaml.safe_load(Path(args.spec).read_text(encoding="utf-8"))
    root = Path(args.asset_root).resolve()
    model_root = root / "models"
    model_root.mkdir(parents=True, exist_ok=True)
    selected = set(args.only)
    manifest_path = root / "model-assets.manifest.json"
    prior = {}
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        prior = {item["key"]: item for item in existing.get("models", [])}
    required = {
        key: item
        for key, item in spec["models"].items()
        if item.get("execution_required", True)
    }
    records = dict(prior)
    if args.refresh_manifest:
        missing = [key for key in required if key not in records]
        if missing:
            raise RuntimeError(
                "cannot refresh an incomplete runtime manifest; missing: " + ", ".join(missing)
            )
    else:
        for key, item in required.items():
            if selected and key not in selected:
                continue
            destination = model_root / key
            source = item.get("source", "huggingface")
            if source == "modelscope":
                verify_modelscope_revision(item)
            command = download_command(item, destination)
            subprocess.run(command, check=True)
            files = []
            for path in sorted(destination.rglob("*")):
                if not path.is_file() or ".cache/huggingface" in path.as_posix():
                    continue
                files.append(
                    {
                        "path": str(path.relative_to(root)),
                        "bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                    }
                )
            records[key] = {
                "key": key,
                "source": source,
                "repo_id": item["repo_id"],
                "revision": item["revision"],
                "purpose": item["purpose"],
                "files": files,
            }
    for key, item in required.items():
        if key not in records:
            continue
        records[key].update(
            {
                "source": item.get("source", "huggingface"),
                "repo_id": item["repo_id"],
                "revision": item["revision"],
                "purpose": item["purpose"],
            }
        )
        if item.get("upstream_repo_id"):
            records[key]["upstream_repo_id"] = item["upstream_repo_id"]
        if item.get("source_revision"):
            records[key]["source_revision"] = item["source_revision"]
    ordered_records = [records[key] for key in required if key in records]
    spec_path = Path(args.spec).resolve()
    manifest = {
        "schema_version": 1,
        "models_spec": "offline/models.yaml",
        "models_spec_sha256": sha256_file(spec_path),
        "models": ordered_records,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {"status": "ok", "asset_root": str(root), "models": len(ordered_records)},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
