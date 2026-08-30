#!/usr/bin/env python3
"""Freeze the executable project surface for offline transfer verification."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT_FILES = ("AGENTS.md", "README.md", "plan.md", "pyproject.toml")
TREE_PATTERNS = {
    "configs": ("*.yaml",),
    "offline": ("*.yaml", "*.txt"),
    "scripts": ("*.py", "*.sh"),
    "skills": ("*.md", "*.py", "*.yaml"),
    "src": ("*.py",),
    "tests": ("*.py",),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output", default="offline/code.manifest.json")
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    output = (root / args.output).resolve()
    paths = [root / value for value in ROOT_FILES]
    for tree, patterns in TREE_PATTERNS.items():
        tree_root = root / tree
        for pattern in patterns:
            paths.extend(tree_root.rglob(pattern))
    selected = sorted(
        {
            path.resolve()
            for path in paths
            if path.is_file()
            and path.resolve() != output
            and "__pycache__" not in path.parts
            and ".pytest_cache" not in path.parts
        }
    )
    records = [
        {
            "path": str(path.relative_to(root)),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in selected
    ]
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project_root_marker": "pyproject.toml",
        "files": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "output": str(output), "files": len(records)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
