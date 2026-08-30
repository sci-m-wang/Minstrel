#!/usr/bin/env python3
"""Hash a prepared Linux wheelhouse for fail-closed offline installation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", required=True)
    parser.add_argument("--requirements", default="offline/requirements-gpu.txt")
    args = parser.parse_args()
    root = Path(args.asset_root).resolve()
    wheelhouse = root / "wheelhouse"
    requirements = Path(args.requirements).resolve()
    if not wheelhouse.is_dir():
        raise FileNotFoundError(wheelhouse)
    files = [
        {
            "path": str(path.relative_to(root)),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(wheelhouse.iterdir())
        if path.is_file()
    ]
    if not files:
        raise RuntimeError("wheelhouse is empty")
    manifest = {
        "schema_version": 1,
        "requirements": "offline/requirements-gpu.txt",
        "requirements_sha256": sha256_file(requirements),
        "files": files,
    }
    output = root / "wheelhouse.manifest.json"
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "output": str(output), "files": len(files)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
