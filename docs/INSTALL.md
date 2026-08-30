# Installation and asset preparation

This repository is the public, model-free experiment package. Formal execution additionally needs
the pinned model asset directory, an offline Linux wheelhouse, and the separately supplied frozen
research corpus. Build or obtain those assets on a connected preparation host; the disconnected GPU
Agent only verifies and executes them.

## 1. Clone and inspect the executable design

```bash
git clone https://github.com/sci-m-wang/Minstrel.git
cd Minstrel
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest
```

The formal scope is fixed in `configs/scope.yaml`. Do not edit baselines, coverage gates, actor
matrices, generation settings, configs, or manifests on the GPU host.

## 2. Prepare model assets on a connected host

Use a Linux x86_64 host compatible with the target's Python 3.11/CUDA runtime. Install the repository
clients in a preparation-only environment:

```bash
python3.11 -m venv .model-download-venv
source .model-download-venv/bin/activate
python -m pip install PyYAML huggingface_hub modelscope
export SIDEPROFILE_ASSET_ROOT=/absolute/path/to/side_profile_offline_assets
python scripts/download_offline_models.py --asset-root "$SIDEPROFILE_ASSET_ROOT"
```

`offline/models.yaml` is authoritative. Accept any upstream gated-model licenses before download.
Llama is a special case: `llama-3.1-8b-instruct` must come from ModelScope repository
`LLM-Research/Meta-Llama-3.1-8B-Instruct`, never Hugging Face. The downloader verifies that the
ModelScope `master` branch resolves to pinned commit
`359efdbb8af05b788a4ad4185215c6b8caa9052c` before downloading and then freezes every retained file
hash. The script does not set retry, worker, quantization, decoding, or model-length parameters.

The prepared directory has this contract:

```text
side_profile_offline_assets/
  model-assets.manifest.json
  models/
    llama-3.1-8b-instruct/
    qwen2.5-7b-instruct/
    qwen2.5-14b-instruct/
    gemma-2-9b-it/
    mistral-7b-instruct-v0.3/
    bge-reranker-v2-m3/
    baichuan-char-rm/
```

Qwen3-Embedding-0.6B is not needed during formal execution because the public repository already
contains the frozen exact-cosine vector database. Rebuilding that database is a preparation-side task
and invalidates the frozen bundle.

## 3. Build the offline GPU wheelhouse

On a connected Linux x86_64/Python 3.11 host compatible with the target CUDA installation:

```bash
SIDEPROFILE_PREP_PYTHON=python3.11 \
  scripts/prepare_remote_wheelhouse.sh . "$SIDEPROFILE_ASSET_ROOT"
```

The script uses the selected Python only to create a dedicated builder environment, downloads the
exact versions in `offline/requirements-gpu.txt`, and freezes their hashes. It assumes no scheduler,
module system, account, partition, or storage layout.

Transfer the repository and the complete asset directory to the disconnected host without changing
their contents.

## 4. Place the restricted frozen corpus

The public repository intentionally omits `data/corpus/comments.sqlite`: most source comments were
retained for private research audit and are not licensed for public redistribution. Obtain the exact
frozen file from the study preparation side and place it at:

```text
data/corpus/comments.sqlite
```

Do not recreate, scrape, translate, repair, or substitute it on the execution host. Its expected hash
and the per-role coverage inventory are recorded in `data/manifests/` and
`data/audits/corpus_inventory.json`. A mismatch is a hard preflight failure.

## 5. Discover and install on the target GPU machine

Before installing, read the target machine's project/site rules. Resolve its scheduler, GPU/account
limits, module/runtime policy, network policy, storage size and file-count quotas, and appropriate
locations for the project, environment, large model files, outputs, and logs. Then follow
`skills/side-profile-experiment/SKILL.md`.

The generic offline environment commands are:

```bash
export SIDEPROFILE_ASSET_ROOT=/absolute/path/to/side_profile_offline_assets
export SIDEPROFILE_VENV=/absolute/writable/path/to/sideprofile-venv
python3.11 -m venv "$SIDEPROFILE_VENV"
source "$SIDEPROFILE_VENV/bin/activate"
python -m pip install --no-index \
  --find-links "$SIDEPROFILE_ASSET_ROOT/wheelhouse" \
  -r offline/requirements-gpu.txt
python -m pip install --no-index --no-build-isolation -e .
python -m pytest
```

Run the target inventory and both fail-closed preflights exactly as described in
`skills/side-profile-experiment/references/operations.md`. Formal vLLM stages must use
`scripts/run_vllm_stage.sh`; it does not set output-token caps, retry limits, temperature, top-p,
seed, dtype, quantization, or model length.
