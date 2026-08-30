#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 || $# -gt 5 ]]; then
  echo "usage: $0 prepare|actor CONFIG MODEL_KEY ASSET_ROOT [PREPARED_DIR]" >&2
  exit 2
fi

stage=$1
config=$2
model_key=$3
asset_root=$4
prepared_dir=${5:-}
model_path="$asset_root/models/$model_key"
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

if [[ ! -d "$model_path" ]]; then
  echo "missing frozen model directory: $model_path" >&2
  exit 2
fi
if [[ "$stage" == actor && -z "$prepared_dir" ]]; then
  echo "actor stage requires PREPARED_DIR" >&2
  exit 2
fi
if [[ "$stage" != prepare && "$stage" != actor ]]; then
  echo "stage must be prepare or actor" >&2
  exit 2
fi

export LOCAL_API_KEY=local-offline
export LOCAL_BASE_URL=http://127.0.0.1:8000/v1
export LOCAL_MODEL=$model_key
export SIDEPROFILE_ASSET_ROOT=$asset_root
export PYTHONPATH="$script_dir/text_runtime${PYTHONPATH:+:$PYTHONPATH}"

log_dir=${SIDEPROFILE_VLLM_LOG_DIR:-runs/vllm-logs}
mkdir -p "$log_dir"
log_path="$log_dir/${stage}-${model_key}-$(date -u +%Y%m%dT%H%M%SZ).log"

python3 -m sideprofile.vllm_launcher serve "$model_path" \
  --served-model-name "$model_key" \
  --host 127.0.0.1 \
  --port 8000 >"$log_path" 2>&1 &
server_pid=$!

stop_server() {
  if kill -0 "$server_pid" 2>/dev/null; then
    kill -TERM "$server_pid"
    wait "$server_pid" || true
  fi
}
trap stop_server EXIT INT TERM

until curl -fsS http://127.0.0.1:8000/health >/dev/null; do
  if ! kill -0 "$server_pid" 2>/dev/null; then
    wait "$server_pid"
    exit $?
  fi
  sleep 2
done

if [[ "$stage" == prepare ]]; then
  sideprofile prepare-conditionings --config "$config"
else
  sideprofile run-actor --config "$config" --prepared-dir "$prepared_dir"
fi
