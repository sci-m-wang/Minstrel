---
name: side-profile-experiment
description: Execute, audit, and analyze a locally frozen SideProfile identity-blind character-comment experiment on a GPU machine. Use for deployment, checksum preflight, 24-probe experiment runs, resumption, official evaluation, or evidence-backed reports. Never use it to collect, import, synthesize, repair, or rebuild the comment corpus or benchmark data.
---

# SideProfile Experiment

Operate a complete, frozen, read-only experimental input bundle through GPU execution and a
reviewable result report. Treat `plan.md` as the design authority, the selected config as the frozen
run specification, and `data.bundle_manifest` as the checksum authority.

The executable scope is defined by `configs/scope.yaml`: Panels A/D and the six conditions `none`,
`personality`, `raw`, `summary`, `gold`, and `ours`. AMADEUS, RoleGPT/RoleLLM, PersonaForge, and
CoSER are removed until pinned external artifacts and tested adapters exist. Do not reintroduce or
claim those baselines merely because they remain in the historical sections of `plan.md`.

## Hard responsibility boundary

The local preparation side owns comment acquisition, privacy filtering, corpus construction,
deduplication, catalog/gold-profile completion, benchmark conversion, source/license audit, model and
wheelhouse download, Qwen3 vector-database construction and verification, config freezing, and local
tests. The GPU-side Agent owns only offline
environment installation from the supplied wheelhouse, frozen-input verification, experiment
execution, official evaluation, analysis, and reporting.

On the GPU machine, never:

- browse, scrape, call a dataset API, download any model/dataset/package, or ask the user to supply comments;
- run `init-corpus`, `import-comments`, a collector, or any command that writes the SQLite corpus;
- run `build-vector-store`, download/load the embedding model, or alter the frozen vector database;
- add, synthesize, translate, relabel, deduplicate, or delete comments;
- edit the catalog, gold profiles, benchmark JSONL, config, executable scope, or bundle manifest;
- replace a missing input, relax coverage, change a baseline, or substitute BM25 for required hybrid retrieval.

Installing declared dependencies only from the supplied offline wheelhouse is allowed. If any input,
model, or wheel is missing, or a checksum/coverage check fails, stop before the first model call and
report the exact failed check to the local preparation side. A preflight block means the bundle is
incomplete or changed; it is a safety stop, not an instruction to repair data on the GPU host.

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

Stop if the preflight reports missing environment keys, inputs, a missing/stale Qwen3 vector
database, missing reranker support for a `hybrid` run, a missing/mismatched frozen bundle, or
synthetic data in a research configuration.
Never display `.env` values. Never fix a data-side failure on this host.

## Route the task

- For environment setup, deployment, smoke tests, full execution, artifact locations, or recovery,
  read [references/operations.md](references/operations.md).
- For scoring, aggregation, statistical comparison, interpretation, or a paper-ready report, read
  [references/analysis-report.md](references/analysis-report.md).

Do not route GPU-side work to `references/data-contract.md`; that file documents work that must
already have been completed locally before the bundle was frozen.

## Required invariants

- Preserve the private mapping from `character_id` to `anonymous_id`; Actor and profiling calls must
  never receive character name, work, official profile, wiki text, or raw unmasked aliases.
- Keep deterministic alias/work masking active on retrieved text and every generated Cue, Person
  Model statement, generated baseline conditioning, and Actor response. A remaining leak after
  masking invalidates the unit; do not add a retry or weaken the leak detector.
- Preserve `Comment → Cue → Person Model`. Every non-unknown Cue must cite imported comment IDs;
  unsupported citations are rejected by code.
- Exclude `is_synthetic=true` by default. A run with `include_synthetic: true` is a smoke test and
  must never be presented as experimental evidence.
- Use all 24 fixed probes for a main run unless the config explicitly identifies an ablation or smoke
  test. Do not silently substitute BM25-only retrieval when the research config requires `hybrid`.
- Read dense scores from the frozen exact-cosine vector database built with the pinned
  `Qwen3-Embedding-0.6B` revision. The database must match the corpus comment IDs/text hashes and all
  24 English/Chinese probe queries. Never rebuild embeddings during an experiment run or replace the
  exact store with an approximate index.
- Freeze actor, provider-default decoding, replicate count, evidence budget, conditioning budget, and evaluator across controlled
  conditions. Report deviations.
- Prepare the shared six conditionings exactly once per panel with the fixed
  `Qwen2.5-14B-Instruct` profiler. Every actor must checksum and reuse that prepared directory; an
  actor must never reconstruct profiles or baseline conditionings.
- Do not introduce API `max_tokens`, call-count budgets, or application retry limits. The planned
  Raw / Summary / Ours conditioning size is 1000±50 tokens measured by the fixed
  Qwen2.5-14B-Instruct profiler tokenizer; it is an experimental treatment
  requirement, not an API generation cap. The reconstructed Person Model's broader design range is
  800–1200 tokens, but the controlled main comparison uses the stricter shared 950–1050 gate.
- Do not set or expose `temperature`, `top_p`, or API `seed`; use the model/vLLM/provider defaults for
  every condition. Repeated trials are independent replicates, not explicitly seeded generations.
- Preserve every retained baseline definition exactly. If an external baseline cannot run from pinned
  official artifacts without methodological substitutions, report it as unavailable; do not remove,
  approximate, or rewrite anything on the GPU host.
- Treat the corpus, vector database, catalog, benchmark, config, scope file, and bundle manifest as
  immutable inputs.
- Inspect `manifest.json`, corpus coverage, Cue citations, identity leakage, failed calls, and official
  evaluator status before interpreting scores.

## Finish criteria

A completed panel has a verified prepared-conditioning directory; one completed actor directory per
pre-registered actor; `generations.jsonl`; the applicable official evaluator artifact; and a final
report. Panel A must label General Response as pending human/GPT-4 pairwise evaluation when offline;
Panel D must contain CharacterRM scores. Report exact paths, hashes, missing cells, and whether every
run is `research_valid`. Do not claim benchmark results from a synthetic smoke run.
