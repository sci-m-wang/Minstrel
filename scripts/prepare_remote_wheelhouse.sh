#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 PROJECT_ROOT ASSET_ROOT" >&2
  exit 2
fi

project_root=$1
asset_root=$2

prep_python=${SIDEPROFILE_PREP_PYTHON:-python3}
if ! command -v "$prep_python" >/dev/null 2>&1; then
  echo "preparation Python not found: $prep_python" >&2
  exit 2
fi

builder_python="$asset_root/wheelhouse-builder/bin/python"
if [[ ! -x "$builder_python" ]]; then
  "$prep_python" -m venv "$asset_root/wheelhouse-builder"
fi
mkdir -p "$asset_root/wheelhouse"
cd "$project_root"
"$builder_python" -m pip download \
  --only-binary=:all: \
  --dest "$asset_root/wheelhouse" \
  --requirement offline/requirements-gpu.txt
"$builder_python" scripts/build_wheelhouse_manifest.py --asset-root "$asset_root"
