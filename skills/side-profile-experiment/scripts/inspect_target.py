#!/usr/bin/env python3
"""Inventory a target GPU machine without changing frozen experiment inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import yaml


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def path_status(path: Path) -> dict:
    probe = path if path.exists() else path.parent
    return {
        "path": str(path),
        "exists": path.exists(),
        "is_file": path.is_file(),
        "is_dir": path.is_dir(),
        "writable_or_parent_writable": probe.exists() and os.access(probe, os.W_OK),
    }


def visible_gpus() -> dict:
    command = shutil.which("nvidia-smi")
    if not command:
        return {"command": None, "visible": False, "devices": []}
    try:
        result = subprocess.run(
            [
                command,
                "--query-gpu=index,name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        return {
            "command": command,
            "visible": False,
            "devices": [],
            "detail": str(exc),
        }
    devices = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return {
        "command": command,
        "visible": result.returncode == 0 and bool(devices),
        "devices": devices,
        "detail": result.stderr.strip(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--asset-root", required=True)
    parser.add_argument("--venv-path", required=True)
    parser.add_argument("--rules-file", action="append", required=True)
    parser.add_argument("--config", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    root = Path(args.project_root).expanduser().resolve()
    assets = Path(args.asset_root).expanduser().resolve()
    venv = Path(args.venv_path).expanduser().resolve()
    failures: list[str] = []

    if not (root / "pyproject.toml").is_file() or not (root / "src/sideprofile").is_dir():
        failures.append("project root is missing pyproject.toml or src/sideprofile")
    if not assets.is_dir():
        failures.append(f"asset root is missing: {assets}")
    if sys.version_info[:2] != (3, 11):
        failures.append(f"Python 3.11 is required; found {platform.python_version()}")
    venv_probe = venv if venv.exists() else venv.parent
    if not venv_probe.exists() or not os.access(venv_probe, os.W_OK):
        failures.append(f"virtual-environment path is not writable: {venv}")

    rules = []
    for value in args.rules_file:
        path = resolve(root, value)
        if not path.is_file():
            failures.append(f"deployment rules file is missing: {path}")
            rules.append({"path": str(path), "exists": False})
        else:
            rules.append(
                {"path": str(path), "exists": True, "sha256": sha256_file(path)}
            )

    registry_path = root / "offline/models.yaml"
    registry = (
        yaml.safe_load(registry_path.read_text(encoding="utf-8"))
        if registry_path.is_file()
        else {}
    ) or {}
    if not registry_path.is_file():
        failures.append(f"model registry is missing: {registry_path}")
    models = []
    for key, item in registry.get("models", {}).items():
        if not item.get("execution_required", True):
            continue
        path = assets / "models" / key
        present = path.is_dir()
        if not present:
            failures.append(f"runtime model is missing: {key} at {path}")
        models.append(
            {
                "key": key,
                "source": item.get("source", "huggingface"),
                "repo_id": item.get("repo_id"),
                "revision": item.get("revision"),
                "path": str(path),
                "present": present,
            }
        )

    data_records: dict[str, dict] = {}
    config_records = []
    for value in args.config:
        config_path = resolve(root, value)
        if not config_path.is_file():
            failures.append(f"config is missing: {config_path}")
            config_records.append({"path": str(config_path), "present": False})
            continue
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        config_records.append(
            {"path": str(config_path), "present": True, "sha256": sha256_file(config_path)}
        )
        data = config.get("data", {})
        for field in ("corpus_db", "catalog", "benchmark", "bundle_manifest", "model_registry"):
            if data.get(field):
                path = resolve(root, str(data[field]))
                data_records[str(path)] = {"field": field, "path": str(path), "present": path.is_file()}
        for value in data.get("frozen_assets", []):
            path = resolve(root, str(value))
            data_records[str(path)] = {"field": "frozen_asset", "path": str(path), "present": path.is_file()}
        vector_value = config.get("retrieval", {}).get("vector_store")
        if vector_value:
            path = resolve(root, str(vector_value))
            data_records[str(path)] = {"field": "vector_store", "path": str(path), "present": path.is_file()}
    for item in data_records.values():
        if not item["present"]:
            failures.append(f"frozen data is missing: {item['path']}")

    output_path = resolve(root, args.output)
    output_probe = output_path.parent
    if not output_probe.exists() or not os.access(output_probe, os.W_OK):
        failures.append(f"deployment inventory output directory is not writable: {output_probe}")

    payload = {
        "schema_version": 1,
        "ok": not failures,
        "failures": failures,
        "machine": {
            "hostname": platform.node(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": {"executable": sys.executable, "version": platform.python_version()},
            "gpus": visible_gpus(),
            "scheduler_commands": {
                name: shutil.which(name) for name in ("sbatch", "srun", "qsub")
            },
        },
        "paths": {
            "project_root": path_status(root),
            "asset_root": path_status(assets),
            "venv_path": path_status(venv),
            "inventory_output": path_status(output_path),
        },
        "rules": rules,
        "configs": config_records,
        "models": models,
        "data": list(data_records.values()),
    }
    if output_probe.exists() and os.access(output_probe, os.W_OK):
        output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
