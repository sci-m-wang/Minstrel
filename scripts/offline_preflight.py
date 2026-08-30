#!/usr/bin/env python3
"""Fail-closed validation for a disconnected GPU experiment host."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

import yaml


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--asset-root", required=True)
    args = parser.parse_args()
    project = Path(args.project_root).resolve()
    assets = Path(args.asset_root).resolve()
    checks = []

    def check(name: str, ok: bool, detail: str) -> None:
        checks.append({"check": name, "ok": ok, "detail": detail})

    check("project", (project / "pyproject.toml").is_file(), str(project))
    code_manifest_path = project / "offline" / "code.manifest.json"
    check("code_manifest", code_manifest_path.is_file(), str(code_manifest_path))
    if code_manifest_path.is_file():
        code_manifest = json.loads(code_manifest_path.read_text(encoding="utf-8"))
        code_failures = []
        for item in code_manifest.get("files", []):
            path = project / item["path"]
            if not path.is_file():
                code_failures.append(f"missing {item['path']}")
            elif path.stat().st_size != item["bytes"]:
                code_failures.append(f"size {item['path']}")
            elif sha256_file(path) != item["sha256"]:
                code_failures.append(f"sha256 {item['path']}")
        check(
            "code_files",
            bool(code_manifest.get("files")) and not code_failures,
            "; ".join(code_failures) or f"{len(code_manifest.get('files', []))} files",
        )
    manifest_path = assets / "model-assets.manifest.json"
    check("model_manifest", manifest_path.is_file(), str(manifest_path))
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        spec_path = project / "offline/models.yaml"
        check(
            "model_spec",
            spec_path.is_file()
            and sha256_file(spec_path) == manifest.get("models_spec_sha256"),
            str(spec_path),
        )
        if spec_path.is_file():
            spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
            expected = {
                key: {
                    "revision": item["revision"],
                    "source": item.get("source", "huggingface"),
                    "repo_id": item["repo_id"],
                    "source_revision": item.get("source_revision"),
                }
                for key, item in spec.get("models", {}).items()
                if item.get("execution_required", True)
            }
            observed = {
                item["key"]: {
                    "revision": item.get("revision"),
                    "source": item.get("source", "huggingface"),
                    "repo_id": item.get("repo_id"),
                    "source_revision": item.get("source_revision"),
                }
                for item in manifest.get("models", [])
            }
            inventory_failures = [
                f"{key}:{observed.get(key, 'missing')}!={expected_item}"
                for key, expected_item in expected.items()
                if observed.get(key) != expected_item
            ]
            inventory_failures.extend(
                f"unexpected:{key}" for key in observed if key not in expected
            )
            check(
                "model_inventory",
                not inventory_failures,
                "; ".join(inventory_failures) or f"{len(expected)} pinned models",
            )
        for model in manifest.get("models", []):
            failures = []
            for item in model.get("files", []):
                path = assets / item["path"]
                if not path.is_file():
                    failures.append(f"missing {item['path']}")
                elif path.stat().st_size != item["bytes"]:
                    failures.append(f"size {item['path']}")
                elif sha256_file(path) != item["sha256"]:
                    failures.append(f"sha256 {item['path']}")
            check(f"model:{model['key']}", not failures, "; ".join(failures) or model["revision"])
    wheel_manifest_path = assets / "wheelhouse.manifest.json"
    check("wheelhouse_manifest", wheel_manifest_path.is_file(), str(wheel_manifest_path))
    if wheel_manifest_path.is_file():
        wheel_manifest = json.loads(wheel_manifest_path.read_text(encoding="utf-8"))
        requirements_path = project / "offline/requirements-gpu.txt"
        check(
            "wheelhouse_requirements",
            requirements_path.is_file()
            and sha256_file(requirements_path)
            == wheel_manifest.get("requirements_sha256"),
            str(requirements_path),
        )
        wheel_failures = []
        for item in wheel_manifest.get("files", []):
            path = assets / item["path"]
            if not path.is_file():
                wheel_failures.append(f"missing {item['path']}")
            elif path.stat().st_size != item["bytes"]:
                wheel_failures.append(f"size {item['path']}")
            elif sha256_file(path) != item["sha256"]:
                wheel_failures.append(f"sha256 {item['path']}")
        check(
            "wheelhouse_files",
            bool(wheel_manifest.get("files")) and not wheel_failures,
            "; ".join(wheel_failures) or f"{len(wheel_manifest.get('files', []))} files",
        )
    for module in ("sideprofile", "vllm", "torch", "transformers", "sentence_transformers"):
        check(f"python:{module}", importlib.util.find_spec(module) is not None, module)
    check(
        "text_vllm_launcher",
        importlib.util.find_spec("sideprofile.vllm_launcher") is not None,
        "sideprofile.vllm_launcher",
    )
    check("vllm_command", shutil.which("vllm") is not None, str(shutil.which("vllm")))
    result = {"ok": all(item["ok"] for item in checks), "checks": checks}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
