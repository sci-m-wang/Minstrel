# Remove Conditioning Length Controls Implementation Plan

> **For agentic workers:** Execute these steps inline in the current task; do not delegate or add experimental conditions.

**Goal:** Remove every artificial conditioning-length target, truncation, and length gate while preserving the six existing conditions and their native information forms.

**Architecture:** Keep the frozen retrieval policy based on probes and comment records. Build one complete deduplicated retrieved-comment rendering for Raw, Summary, and Personality; allow Summary, Personality, and Ours to generate their native outputs without requested lengths. Keep Gold unchanged and keep Cue extraction internal to Ours.

**Tech Stack:** Python 3.11, pytest, YAML configs, Markdown Skill/docs, SQLite bundle manifests.

---

### Task 1: Specify native-length behavior in tests

**Files:**
- Modify: `tests/test_pipeline_mock.py`

- [x] Replace the Raw token-budget test with a test asserting that every unique retrieved comment is retained exactly once.
- [x] Add a recording mock test asserting that Summary and Personality receive the complete deduplicated comment rendering and that their prompts contain no requested token length.
- [x] Run `pytest tests/test_pipeline_mock.py -q`; expect the new tests to fail against the old truncating implementation.

### Task 2: Remove length controls from the pipeline

**Files:**
- Modify: `src/sideprofile/pipeline.py`
- Modify: `src/sideprofile/profile.py`
- Modify: `src/sideprofile/cli.py`
- Modify: `src/sideprofile/staged.py`

- [x] Remove `target_tokens`, token counters, clipping helpers, requested output ranges, tokenizer loading, length artifacts, and length validation.
- [x] Render the complete deduplicated retrieved comment set for Raw and as the source input to Summary and Personality.
- [x] Keep Ours as `comments -> per-probe Cues -> Person Model`; send the Person Model, not raw Cues, to Actor.
- [x] Run `pytest tests/test_pipeline_mock.py tests/test_staged.py -q`; expect all selected tests to pass.

### Task 3: Remove the rejected design from configs, audits, and reports

**Files:**
- Modify: `configs/*.yaml`
- Modify: `configs/offline/*.yaml`
- Modify: `scripts/build_experiment_audit.py`
- Modify: `src/sideprofile/report.py`
- Modify: `plan.md`

- [x] Delete all `target_tokens`, conditioning targets, accepted ranges, and token-budget comparison language.
- [x] Describe Raw, Summary, Personality, Ours, Gold, and None as distinct internal conditions/sub-experiments; describe main experiments as comparisons with other methods.
- [x] Rebuild `data/audits/experiment_inventory.json` using `scripts/build_experiment_audit.py` and confirm it contains no conditioning-length fields.

### Task 4: Update and validate the execution Skill

**Files:**
- Modify: `skills/side-profile-experiment/SKILL.md`
- Modify: `skills/side-profile-experiment/references/analysis-report.md`
- Modify: `skills/side-profile-experiment/scripts/preflight.py`

- [x] Remove the 1000±50 invariant and any instruction to normalize condition lengths.
- [x] Make a configured conditioning-length target or truncation setting a preflight failure.
- [x] State that each condition keeps its native payload and Gold is never length-normalized.
- [x] Run the Skill quick validator; expect `Skill is valid!`.

### Task 5: Re-freeze and publish

**Files:**
- Regenerate: `offline/code.manifest.json`
- Regenerate: `data/manifests/panel-a.json`
- Regenerate: `data/manifests/panel-d.json`

- [x] Run the complete local pytest suite; expect all tests to pass.
- [x] Rebuild the experiment audit, code manifest, and both bundle manifests, then verify both bundles.
- [x] Copy and validate the installed Skill at `~/.codex/skills/side-profile-experiment`.
- [x] Commit and push the corrected frozen package to the private `sci-m-wang/Minstrel` repository.
