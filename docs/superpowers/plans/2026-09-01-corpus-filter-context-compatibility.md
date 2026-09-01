# Corpus Filter and Context Compatibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Freeze a reproducibly cleaned 8,673-row comment corpus and prove that every planned model receives inputs within its native context window without truncation or model-length overrides.

**Architecture:** The local preparation side builds a new SQLite database from the current checksum-verified source, applying two explicit exclusion rules: Stack Exchange questions/answers and all comments with at least 1,000 Unicode characters. It rebuilds all dependent audits, the Qwen3 exact vector store, code/data manifests, and deployment Skill. The SMU preparation/validation run inventories native model contexts, removes any Actor with a native window of 8K or less, prepares the five conditions once per panel, tokenizes/audits every planned input, and validates the smallest retained Actor with real vLLM traces.

**Tech Stack:** Python 3.11, SQLite, Pydantic, pytest, Qwen3-Embedding-0.6B, exact-cosine SQLite vector store, Transformers tokenizers, vLLM, Slurm/SMU cluster.

---

### Task 1: Add a deterministic corpus filter

**Files:**
- Create: `scripts/filter_comment_corpus.py`
- Create: `tests/test_filter_comment_corpus.py`

- [x] **Step 1: Write failing tests**

Test that the filter refuses to overwrite its input or an existing output, preserves schema/metadata/characters, excludes Stack Exchange question and answer URLs, keeps Stack Exchange `#comment` rows below 1,000 characters, excludes every remaining row with `LENGTH(raw_text) >= 1000`, preserves synthetic fixtures unchanged, and writes before/after counts plus SHA-256 hashes.

- [x] **Step 2: Run the focused test and confirm failure**

Run: `PYTHONPATH=src python3 -m pytest tests/test_filter_comment_corpus.py -q`

Expected: import failure because the filter script does not yet exist.

- [x] **Step 3: Implement copy-then-filter**

Open the source database read-only, copy it through SQLite backup into a new path, execute only schema-qualified `DELETE FROM main.comments ...` statements on the new database, checkpoint/VACUUM the output, run `PRAGMA integrity_check`, and emit a JSON audit keyed by stable comment IDs. Never issue an unqualified destructive SQL statement against an attached database.

- [x] **Step 4: Run focused and full tests**

Run: `PYTHONPATH=src python3 -m pytest tests/test_filter_comment_corpus.py -q`

Run: `PYTHONPATH=src python3 -m pytest -q`

Expected: all tests pass.

### Task 2: Build and validate the cleaned corpus

**Files:**
- Replace after verification: `data/corpus/comments.sqlite`
- Preserve privately: `data/private/recovery/pre-filter-comments.sqlite`
- Create: `data/audits/comment_filter_audit.json`
- Regenerate: `data/audits/comment_length_distribution.json`
- Regenerate: `data/audits/corpus_inventory.json`

- [x] **Step 1: Build to a new filename**

Run the filter against the current corpus and require 10,540 source rows, 1,787 source-structure exclusions, 80 additional length exclusions, and 8,673 retained non-synthetic rows.

- [x] **Step 2: Verify data quality**

Require unique `comment_id`, unique `(character_id, text_hash)`, zero empty text, 16 expected characters, at least two platforms and 100 authors per formal character, maximum retained text length 999, and no retained Stack Exchange question/answer URL.

- [x] **Step 3: Atomically promote the verified database**

Move the old checksum-verified corpus into the private recovery directory and rename the verified new database to `data/corpus/comments.sqlite`. Preserve the old file; do not delete it.

- [x] **Step 4: Regenerate audits**

Run `scripts/build_comment_length_audit.py` and `scripts/build_corpus_audit.py`. Treat comment counts as descriptive sample sizes, not a 500-row eligibility threshold.

### Task 3: Remove obsolete review and coverage rules

**Files:**
- Delete: `scripts/review_comment_corpus.py`
- Delete: `tests/test_review_comment_corpus.py`
- Delete: `docs/superpowers/plans/2026-08-31-comment-corpus-content-review.md`
- Modify: `configs/offline/panel-a.yaml`
- Modify: `configs/offline/panel-d.yaml`
- Modify: `configs/research.example.yaml`
- Modify: `configs/panel-d.example.yaml`
- Modify: `scripts/build_corpus_audit.py`
- Modify: `plan.md`
- Modify: `README.md`
- Modify: `AGENTS.md`
- Modify: `skills/side-profile-experiment/references/data-contract.md`

- [x] **Step 1: Remove the abandoned GPT review path**

Delete the unshipped review script/tests/plan so another Agent cannot mistake them for required corpus construction.

- [x] **Step 2: Make sample size descriptive**

Set formal config `min_comments` to 1 as a technical non-empty check while retaining two-platform and 100-author integrity checks. Document actual frozen counts and the two deterministic filters; do not introduce a replacement scientific cutoff.

- [x] **Step 3: Correct executable scope documentation**

Replace the stale six-condition/Raw text in `AGENTS.md` with the exact five current conditions.

### Task 4: Inventory and prune model contexts

**Files:**
- Create: `scripts/audit_model_context_windows.py`
- Create: `tests/test_model_context_windows.py`
- Modify after measured inventory: `offline/models.yaml`
- Modify after measured inventory: `skills/side-profile-experiment/references/operations.md`
- Modify after measured inventory: `plan.md`

- [x] **Step 1: Test native context extraction**

Cover `max_position_embeddings`, `n_positions`, `seq_length`, and `max_seq_len`; require a positive native value from each model `config.json` and report tokenizer metadata separately without using it to inflate the native model limit.

- [ ] **Step 2: Inventory every planned model on SMU**

Report native contexts for every local Actor, Qwen3 embedding model, reranker, and CharacterRM. The fixed GPT-5.5 profiler runs only on the connected preparation side, so record its authoritative API usage trace rather than treating it as a local model asset. Distinguish Actor context from per-document retrieval context and CharacterRM's official 4,096-token evaluation procedure.

- [x] **Step 3: Remove undersized Actors**

If an Actor's native context is at most 8,192 tokens, remove it from the actor matrix and model registry rather than truncating inputs or changing vLLM model length. Do not replace it unless an already available, pinned, methodologically equivalent Actor with at least 32,768 native tokens exists.

Gemma-2-9B-it was explicitly removed without replacement. The retained Panel A Actor matrix is Llama-3.1-8B-Instruct, Qwen2.5-7B-Instruct, Qwen2.5-14B-Instruct, and Mistral-7B-Instruct-v0.3; Panel D retains its three registered Actors.

- [ ] **Step 4: Regenerate model/code audits**

Update the operations reference and experiment inventory to match the measured retained matrix.

### Task 5: Rebuild the Qwen3 vector store on the preparation GPU

**Files:**
- Replace after verification: `data/vector_store/qwen3-embedding-0.6b.sqlite`

- [ ] **Step 1: Read SMU rules and inventory paths**

After login recovery, read the actual cluster/site rules and resolve the existing project, model, environment, output, and scheduler paths. Do not copy a previous machine convention into the Skill.

- [ ] **Step 2: Sync the cleaned preparation inputs**

Copy the new corpus and code to a separate preparation directory. Do not edit the downstream execution bundle in place.

- [ ] **Step 3: Build to a new vector-store path**

Use the pinned Qwen3-Embedding-0.6B revision and exact-cosine builder. Do not download models or overwrite the old vector store.

- [ ] **Step 4: Verify and return the new vector store**

Require exact comment-ID/text-hash coverage for all 8,673 rows and all English/Chinese probe queries, then copy the verified store back to the local preparation workspace.

### Task 6: Freeze and preflight the new bundle

**Files:**
- Regenerate: `data/audits/experiment_inventory.json`
- Regenerate: `offline/code.manifest.json`
- Regenerate: `data/manifests/panel-a.json`
- Regenerate: `data/manifests/panel-d.json`

- [ ] **Step 1: Run all local tests and audits**

Require the full pytest suite, corpus filter audit, corpus inventory, experiment inventory, and Skill validator to pass.

- [ ] **Step 2: Freeze both panels**

Build and verify Panel A/D manifests only after the cleaned corpus and rebuilt vector store are in place.

- [ ] **Step 3: Run both local preflights**

Require exact five-condition scope, frozen hashes, non-synthetic data, hybrid retrieval, and the revised descriptive coverage contract.

### Task 7: Audit real input lengths on SMU

**Files:**
- Create under run outputs: profiler and Actor context-audit JSON artifacts

- [ ] **Step 1: Prepare Panel A and Panel D conditions**

Run the fixed GPT-5.5 profiler on the connected local preparation side with provider defaults. Cue, Summary, and Personality must each process one probe's Top-10 independently; aggregation requests receive only derived outputs. Freeze the resulting prepared directory so the disconnected GPU Agent only verifies and loads it.

- [ ] **Step 2: Audit every preparation request**

Use actual vLLM prompt/completion usage and require total tokens below Qwen2.5-14B's measured native context for per-probe Cue/Summary/Personality, their aggregations, and Person Model.

- [ ] **Step 3: Tokenize every planned Actor input**

For every retained Actor tokenizer, render the exact system/user chat template over every benchmark example, five conditions, and prepared payload. Report maximum prompt tokens by model/panel/condition and require positive remaining native context without truncation.

- [ ] **Step 4: Run the smallest retained Actor end to end**

Run one complete five-condition Actor unit using the smallest retained native context, then audit actual prompt/completion/total tokens from its vLLM trace. If any request fails, preserve the evidence and return to local preparation; do not adjust vLLM context or condition payloads.

### Task 8: Publish only after validation

**Files:**
- Update installed Skill after pass: `~/.codex/skills/side-profile-experiment/`
- Sync private repository: `sci-m-wang/Minstrel`

- [ ] **Step 1: Install the validated Skill**

Install the project Skill only after the new bundle and remote context audits pass.

- [ ] **Step 2: Sync and test the private clone**

Copy corpus, vector store, code, configs, audits, manifests, and Skill; exclude `.env`, author salt, private traces, recovery files, models, prepared outputs, and run outputs.

- [ ] **Step 3: Commit and push**

Run tests, manifest verification, secret scan, confirm the GitHub repository remains private, then commit and push. Do not push an unvalidated or stale-vector bundle.
