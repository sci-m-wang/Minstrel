---
name: side-profile-experiment
description: Use when deploying, auditing, executing, or reporting the frozen SideProfile experiment or its official external baseline comparisons on a GPU machine.
---

# SideProfile Experiment

Operate a complete, frozen, read-only experimental input bundle through GPU execution and a
reviewable result report. Treat `plan.md` as the design authority, the selected config as the frozen
run specification, and `data.bundle_manifest` as the checksum authority.

The hard-gated executable scope is defined by `configs/scope.yaml`: Panels A/D and the five Ours
conditions `none`, `personality`, `summary`, `gold`, and `ours`. These are internal conditions and
ablations, not external baselines. AMADEUS, RoleGPT/RoleLLM, PersonaForge, and CoSER form a separate
best-effort external baseline lane. Reserve the resources required by the core, then attempt each
available official implementation unchanged using only remaining capacity or after the core run. A
baseline failure or missing artifact must be recorded and must not delay the core experiment, change
its `research_valid` status, or be replaced by an approximation.

## Hard responsibility boundary

The local preparation side owns comment acquisition, privacy filtering, corpus construction,
deduplication, catalog/gold-profile completion, benchmark conversion, source/license audit, model and
wheelhouse download, connected `text-embedding-3-small` exact-vector construction, connected
`Cohere-rerank-v4.0-pro` candidate reranking, GPT-5.6 Sol conditioning preparation,
prepared-directory freezing, config freezing, and local tests. The
GPU-side Agent owns only offline environment installation from the supplied wheelhouse, frozen-input
and prepared-directory verification, static Actor-input token audits, core experiment execution,
best-effort execution of already supplied official baseline artifacts, official evaluation, analysis,
and reporting.

On the GPU machine, never:

- browse, scrape, call a dataset API, download any model/dataset/package, or ask the user to supply comments;
- run `init-corpus`, `import-comments`, a collector, or any command that writes the SQLite corpus;
- run `build-vector-store`, call an embedding/reranking endpoint, or alter the frozen vector database;
- run `prepare-conditionings`, read/copy a GPT `.env`, call the Profiler, or reconstruct any profile;
- add, synthesize, translate, relabel, deduplicate, or delete comments;
- edit the catalog, gold profiles, benchmark JSONL, config, executable scope, or bundle manifest;
- replace a missing input, relax coverage, change a baseline, or substitute another retrieval method;
- make a baseline runnable by editing its method, prompts, data, profiles, checkpoints, or native
  evaluation procedure.

Installing declared core dependencies only from the supplied offline wheelhouse is allowed. If any
core input, model, or wheel is missing, or a checksum/coverage check fails, stop before the first core
model call and report the exact failed check to the local preparation side. A core preflight block
means the bundle is incomplete or changed; it is a safety stop, not an instruction to repair data on
the GPU host. Missing optional baseline artifacts affect only that baseline's attempt status.

## Locate and preflight

Before installing anything, read the target machine's applicable project/site rules and inventory its
deployment contract. Confirm the available GPU types and allocation mechanism, GPU/account limits,
Python/module policy, network policy, storage quotas and file-count rules, and the approved locations
for the project, fragmented environment files, large model assets, prepared outputs, logs, and final
results. Then verify that the frozen data and every runtime model are already present at those
resolved locations. Do not copy conventions from a previous validation machine. For the commands and
the machine-readable inventory, read [references/operations.md](references/operations.md).

Find the project root containing both `pyproject.toml` and `src/sideprofile`. After target discovery,
run:

```bash
python3 scripts/offline_preflight.py --project-root . --asset-root "$SIDEPROFILE_ASSET_ROOT"
LOCAL_API_KEY=local-offline LOCAL_MODEL=preflight \
  python3 skills/side-profile-experiment/scripts/preflight.py --project-root . --config <config.yaml>
```

Stop if the preflight reports missing environment keys, inputs, a missing/stale
`text-embedding-3-small` vector database, a missing/mismatched frozen bundle or supplied
prepared directory, or synthetic data in a research configuration.
Never display `.env` values. Never fix a data-side failure on this host.

## Route the task

- For environment setup, deployment, smoke tests, full execution, artifact locations, or recovery,
  read [references/operations.md](references/operations.md).
- For AMADEUS, RoleGPT/RoleLLM, PersonaForge, or CoSER discovery and best-effort execution, read
  [references/external-baselines.md](references/external-baselines.md).
- For scoring, aggregation, statistical comparison, interpretation, or a paper-ready report, read
  [references/analysis-report.md](references/analysis-report.md).

Do not route GPU-side work to `references/data-contract.md`; that file documents work that must
already have been completed locally before the bundle was frozen.

## Required invariants

- Preserve the private mapping from `character_id` to `anonymous_id`; Actor and profiling calls must
  never receive character name, work, official profile, wiki text, or raw unmasked aliases.
- Keep deterministic alias/work masking active on retrieved text and every generated Cue, Person
  Model statement, generated internal-condition payload, and Actor response. A remaining leak after
  masking invalidates the unit; do not add a retry or weaken the leak detector.
- Preserve `Comment → Cue → Person Model`. Every non-unknown Cue must cite imported comment IDs;
  unsupported citations are rejected by code.
- Exclude `is_synthetic=true` by default. A run with `include_synthetic: true` is a smoke test and
  must never be presented as experimental evidence.
- Use all 24 fixed probes for a formal Ours run unless the config explicitly identifies an ablation or smoke
  test. Formal retrieval is the frozen `vector_rerank` contract; do not substitute another method.
- Treat the exact-cosine database built locally through `text-embedding-3-small` as provenance-only
  GPU input. It must match every corpus comment ID/text hash and all 24 English/Chinese probe queries.
  Never rebuild embeddings, call Cohere, or rerun retrieval on the GPU host.
- Require the prepared retrieval records to reflect exact-vector Top-20 candidate recall followed by
  complete `Cohere-rerank-v4.0-pro` scoring and local Top-10 selection for every probe. The connected
  preparation request sets neither Cohere `top_n` nor a per-document token limit.
- Preserve the configured retrieval and recorded comment-ID sets. Cue extraction, Summary, and
  Personality must process each probe's Top-10 independently; no such request may contain another
  probe's raw comments. Summary and Personality aggregate only their 24 local outputs. Ours aggregates
  evidence-cited Cues into its Person Model. Gold keeps the benchmark or method profile unchanged;
  None receives no persona information. Do not normalize information content or length across these
  distinct conditions.
- Require the supplied shared five conditionings to have been prepared exactly once per panel with
  the fixed connected-preparation Profiler `GPT/gpt-5.6-sol`. Verify its provider/model, input-bundle
  hash, artifacts, five conditions, per-probe processing contract, and checksums. Every Actor must
  reuse that directory unchanged; the GPU Agent must never reconstruct profiles or
  internal-condition payloads.
- Do not introduce API `max_tokens`, call-count budgets, application retry limits, prompt-level
  length requests, conditioning truncation, or length normalization. Native model-context
  compatibility checks and token-count reporting are required but must not modify the evidence.
  Summary and Personality generate naturally from isolated per-probe inputs and aggregate their local
  outputs; Ours constructs its Person Model naturally from the per-probe Cue pipeline; Gold remains
  the unmodified supplied profile.
- Do not set or expose `temperature`, `top_p`, or API `seed`; use the model/vLLM/provider defaults for
  every condition. Repeated trials are independent replicates, not explicitly seeded generations.
- Preserve every external comparison method exactly. After the core preflights pass and its resources
  are reserved, attempt every official baseline artifact present on the target, in an isolated
  environment and output directory. Record `attempt-status.json` whether it completes, fails, or is
  unavailable. Continue to the next baseline after a failure. Never approximate, rewrite, or silently
  omit the attempt.
- Keep baseline availability separate from core validity. Only a completed official baseline run with
  pinned provenance, native inputs, and its official metric is eligible for a cross-method table.
  Missing or failed baselines remain explicit unavailable/failed rows and do not invalidate Ours.
- Treat the corpus, vector database, catalog, benchmark, config, scope file, and bundle manifest as
  immutable inputs.
- Open frozen SQLite inputs through read-only immutable connections. A preflight or Actor-side read
  must not create WAL/SHM sidecars or write catalog metadata back into the corpus. Gold is read only
  from the checksum-covered panel catalog, never from cached corpus character payloads.
- Inspect `manifest.json`, corpus coverage, Cue citations, identity leakage, failed calls, and official
  evaluator status before interpreting scores. Before the first Actor call, use every retained
  Actor's exact local tokenizer/chat template to count every supplied Panel A/D input over all five
  conditions and require positive native context remaining. After execution, audit every Actor trace
  and report the maximum observed tokens and minimum native context remaining by request stage.

## Finish criteria

A completed core panel has a verified prepared-conditioning directory; one completed actor directory
per pre-registered actor; `generations.jsonl`; the applicable official evaluator artifact; and a final
report. Panel A must label General Response as pending human/GPT-4 pairwise evaluation when offline;
Panel D must contain CharacterRM scores. The final report must also list one baseline attempt status
for each planned external method. Report exact paths, hashes, missing cells, and whether every core
run is `research_valid`. Do not claim benchmark results from a synthetic smoke run or a failed,
modified, or provenance-unknown baseline.
