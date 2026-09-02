# Offline GPU Deployment and Operation

Read this reference for deployment, preflight, staged execution, official evaluation, or recovery.
The connected preparation side has already supplied the repository, frozen data manifests, two
checksum-verified GPT-5.6 Sol prepared-conditioning directories, model assets, and a Linux/Python 3.11
wheelhouse. The GPU Agent must not use a network package index, Hugging Face Hub, or a GPT `.env`.

## Discover the target deployment contract first

Do not assume paths, partitions, accounts, GPU names, quotas, module commands, or network behavior
from another machine. Before installation or the first model call:

1. Read the applicable project instructions and target-site documentation supplied on that machine.
2. Confirm the scheduler/allocation method, allowed GPU count and types, Python/runtime mechanism,
   network policy, storage size and file-count quotas, and which storage class is intended for large
   model files versus fragmented environment/output files.
3. Resolve the project root, model-asset root, virtual-environment path, and the writable project
   location that will contain `runs/`, logs, and reports. The supplied prepared directories are
   immutable inputs, not writable output locations.
4. Run the inventory below. Save its JSON with the run artifacts and stop if `ok` is false.

```bash
python3 skills/side-profile-experiment/scripts/inspect_target.py \
  --project-root <project-root> \
  --asset-root <asset-root> \
  --venv-path <planned-venv-path> \
  --rules-file <applicable-project-or-site-rules> \
  --config configs/offline/panel-a.yaml \
  --config configs/offline/panel-d.yaml \
  --output runs/deployment-inventory.json
```

The inventory records paths, rule-file hashes, Python/platform information, visible GPUs, available
scheduler commands, every frozen data input, and every execution-required model revision/source. It
never reads or prints API secrets. If the target has several rule sources, repeat `--rules-file`.
If written site rules are unavailable, create a short output-side note containing the rules supplied
by the user or administrator and pass that file; do not invent a deployment convention.

`llama-3.1-8b-instruct` is a required actor. Its frozen manifest must identify source `modelscope`,
repository `LLM-Research/Meta-Llama-3.1-8B-Instruct`, and revision
`359efdbb8af05b788a4ad4185215c6b8caa9052c`. ModelScope CLI downloads the `master` source revision,
but the preparation script first requires that branch to resolve to the exact pinned commit and then
freezes every downloaded file hash. It must not be fetched from Hugging Face. Download is a connected
preparation-side action only; a missing Llama directory on the execution host is a hard preflight
failure, not permission for that Agent to download it.

## Immutable inputs and writable outputs

The project root must contain `pyproject.toml`, `configs/offline/`, `data/`, `offline/`, `scripts/`,
and `src/sideprofile/`. Set the supplied model-asset directory:

```bash
export SIDEPROFILE_ASSET_ROOT=/absolute/path/to/side_profile_offline_assets
```

Treat the following as read-only: `data/` (including the frozen `text-embedding-3-small` vector database), `configs/`,
`offline/models.yaml`, both supplied prepared directories, and
`$SIDEPROFILE_ASSET_ROOT/models/`. Only the virtual environment, `runs/`, scheduler logs, and final
report directory are writable during GPU execution.
Frozen SQLite files are opened with the project's immutable read-only path. Their parent directory
does not need write permission and execution must not create `-wal` or `-shm` sidecars.

The supplied `offline/code.manifest.json` freezes the executable source, scripts, tests, configs, and
this Skill. `scripts/offline_preflight.py` verifies every listed file before model startup. Never edit
the code on the GPU host to make a failed preflight pass.

Use the target machine's supplied Python 3.11 runtime. Set `SIDEPROFILE_VENV` to a writable location
chosen for that machine; do not assume a scheduler, module system, home layout, or scratch layout:

```bash
export SIDEPROFILE_VENV=/absolute/writable/path/to/sideprofile-venv
python3 -m venv "$SIDEPROFILE_VENV"
source "$SIDEPROFILE_VENV/bin/activate"
python3 -m pip install --no-index \
  --find-links "$SIDEPROFILE_ASSET_ROOT/wheelhouse" \
  -r offline/requirements-gpu.txt
python3 -m pip install --no-index --no-build-isolation -e .
python3 -m pytest
```

If any wheel is missing or incompatible, stop. Do not connect to PyPI, change dependency versions,
or build a substitute environment on the compute node.

Do not run `build-vector-store`, call the OpenAI-compatible embedding endpoint, or call Cohere on
this execution host. Exact-vector Top-20 recall, complete Cohere candidate reranking, local Top-10
selection, and all five conditionings were completed on the connected preparation side. GPU Actor
execution reads only the immutable prepared payloads. The checksum-covered vector database and
rerank trace are provenance inputs; no embedding or reranking runtime is installed beside vLLM.

## Two mandatory preflights

First verify every model file and required runtime, then verify the frozen experiment bundle:

```bash
python3 scripts/offline_preflight.py \
  --project-root . --asset-root "$SIDEPROFILE_ASSET_ROOT"

LOCAL_API_KEY=local-offline LOCAL_MODEL=preflight python3 skills/side-profile-experiment/scripts/preflight.py \
  --project-root . --config configs/offline/panel-a.yaml
LOCAL_API_KEY=local-offline LOCAL_MODEL=preflight python3 skills/side-profile-experiment/scripts/preflight.py \
  --project-root . --config configs/offline/panel-d.yaml
```

Stop before model startup if any check is false. Return the JSON failure to the preparation side; do
not repair data, fetch a missing asset, lower coverage, or edit a manifest.

Verify the two preparation-side artifacts before any tokenizer or model is loaded:

```bash
sideprofile verify-conditionings --prepared-dir <supplied-panel-a-prepared-dir>
sideprofile verify-conditionings --prepared-dir <supplied-panel-d-prepared-dir>
```

Both manifests must identify `profiler_provider: GPT`, `profiler_model: gpt-5.6-sol`,
`embedding_model: text-embedding-3-small`, `reranker_model: Cohere-rerank-v4.0-pro`, candidate
Top-20/final Top-10, all five conditions, `isolated_per_probe_top10`, the current frozen input-bundle
hash, and valid artifact checksums. A
failure is returned unchanged to the preparation side. Never run `prepare-conditionings` on this
host.

## vLLM execution invariant

Use `scripts/run_vllm_stage.sh`, which sets only the local endpoint/model identity and vLLM network
address. It does not set `max_tokens`, model length, retry policy, temperature, top-p, seed, dtype,
quantization, or another decoding parameter. The model and vLLM defaults therefore remain in force.
The experiment also sets no prompt-level output-length request, conditioning truncation, or length
normalization. Native context-window compatibility is audited without modifying evidence. Each
internal condition keeps its native payload.
The script enters vLLM through the project's text-only launcher, which marks Transformers' optional
torchvision backend unavailable before importing vLLM. This prevents an unused native image
extension from affecting text execution; it does not alter text weights, prompts, or inference
parameters.

Every actor reuses its panel's exact prepared directory. Panel A actor matrix:

```text
llama-3.1-8b-instruct
qwen2.5-7b-instruct
qwen2.5-14b-instruct
mistral-7b-instruct-v0.3
```

Panel D uses the three actors pre-registered by the frozen executable design:

```text
llama-3.1-8b-instruct
qwen2.5-7b-instruct
qwen2.5-14b-instruct
```

Before starting vLLM, statically tokenize every exact Actor request with each retained model's own
local tokenizer and chat template. For each Panel A Actor above, run:

```bash
python3 scripts/audit_actor_input_lengths.py \
  --config configs/offline/panel-a.yaml \
  --prepared-dir <supplied-panel-a-prepared-dir> \
  --model-dir "$SIDEPROFILE_ASSET_ROOT/models/<model-key>" \
  --model-key <model-key> \
  --output "runs/panel-a-<model-key>-input-audit.json"
```

Repeat for each Panel D Actor with the Panel D config and supplied prepared directory. Every audit
must report `ok: true`, no failures, and positive `minimum_remaining_context`. This renders all five
condition payloads without truncation and makes no generation call. A failure is a hard stop: do not
shorten or normalize any input, change model context settings, or silently choose another Actor.

For each model key, run:

```bash
scripts/run_vllm_stage.sh actor configs/offline/<panel>.yaml \
  <model-key> "$SIDEPROFILE_ASSET_ROOT" <panel-prepared-dir>
```

One actor run covers all five conditions and three independent replicates. Do not split or regenerate
individual failed cells. Preserve any failed run directory, then rerun the whole frozen actor unit in
a new directory and disclose the failure.

After the static audits pass, exercise the registered Actor with the smallest native context first.
After each Actor
run, audit its trace with `scripts/audit_context_usage.py --model-dir <actor-model-dir> --trace
<actor-run>/trace.jsonl --output <actor-run>-context-audit.json`. If the smallest-context actor cannot
fit a native five-condition payload, do not change vLLM settings or edit the GPU-side registry. Return
the evidence to the preparation side; a user-authorized replacement with a larger-context registered
model requires a new local registry, config, code manifest, and bundle freeze.

## Official evaluators

Panel A Reaction uses exact official answer accuracy:

```bash
python3 scripts/evaluate_roleagentbench.py \
  --run-dir <panel-a-actor-run> --benchmark data/benchmarks/panel_a.jsonl
```

RoleAgentBench General Response was officially evaluated by human/GPT-4 pairwise comparison. On a
disconnected host, keep those generations and mark the official evaluation pending. Never substitute
a local model judge and call it the official result.

Panel D uses the bundled official CharacterRM. First create its evaluator-only, identity-unmasked
input, then score it locally:

```bash
python3 scripts/prepare_charactereval_run.py --run-dir <panel-d-actor-run>
python3 scripts/evaluate_charactereval.py \
  --run-dir <panel-d-actor-run> \
  --rm-path "$SIDEPROFILE_ASSET_ROOT/models/baichuan-char-rm"
```

The evaluator's official 4096-token left truncation is part of CharacterEval's scoring procedure; it
is not an actor generation limit and must not be altered.

After all pre-registered actors for a panel have their official evaluator artifact, run the strict
panel-level aggregator. Supply every actor run directory exactly once:

```bash
python3 scripts/analyze_panel_results.py --panel A \
  --run-dir <panel-a-llama-run> --run-dir <panel-a-qwen7-run> \
  --run-dir <panel-a-qwen14-run> \
  --run-dir <panel-a-mistral-run> --output-dir results/panel-a-official

python3 scripts/analyze_panel_results.py --panel D \
  --run-dir <panel-d-llama-run> --run-dir <panel-d-qwen7-run> \
  --run-dir <panel-d-qwen14-run> \
  --output-dir results/panel-d-official
```

The command checks actor completeness, exact generation cells, one shared prepared-conditioning
hash, official score availability, and exact paired cells. It writes `condition-summary.csv`,
`paired-contrasts.csv`, `analysis.json`, and `report.md`. It uses paired normal-approximation 95%
intervals, so it introduces neither random resampling nor a statistical seed. A nonzero exit means
the aggregate is incomplete and must not be reported as a full-panel result.

## Analysis and recovery

After official evaluation, read `references/analysis-report.md`. Treat missing cells as missing, not
zero. Match comparisons on character, question, replicate, actor, and frozen prepared-condition hash.
Report exact model revisions from `offline/models.yaml`, bundle hashes, prepared manifest hashes,
failed calls, wall time, corpus coverage, and evaluator status.

Do not analyze a run as research evidence unless its `status.json` is completed, `research_valid` is
true, bundle, vector-database, and prepared checksums pass, the frozen vector-recall/Cohere-rerank
records are complete, all 24 probes
were used, and the claimed official evaluator artifact exists.

## Non-blocking external baseline lane

After the core preflights pass and its compute is reserved, read `references/external-baselines.md`
and attempt AMADEUS, RoleGPT/RoleLLM, PersonaForge, and CoSER from any official artifacts already
supplied on the target. Use only spare capacity or run them after the core. Keep their environments,
processes, logs, and outputs isolated from the frozen core. Missing assets, dependency failures, and
runtime failures are recorded per method in `attempt-status.json` and do not block or invalidate
Panel A/D. Never change a baseline to make it run and never replace a failed baseline with the
internal synthetic smoke configuration.
