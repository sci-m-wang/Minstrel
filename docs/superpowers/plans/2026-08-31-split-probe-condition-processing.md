# Split-Probe Condition Processing Implementation Plan

> **For agentic workers:** Execute this plan inline in the current task. Do not delegate, add context limits, truncate evidence, or alter the frozen retrieval policy.

**Goal:** Remove the non-executable Raw condition and make Cue, Summary, and Personality processing consume each probe's Top-10 independently before condition-native aggregation.

**Architecture:** Retrieval remains BM25 Top-20 plus dense Top-20, RRF, rerank, and Top-10 per each of 24 probes. Ours keeps its existing per-probe Cue calls followed by one Person Model call. Summary and Personality gain one local call per probe followed by one aggregation call over local outputs; no call receives the 24-probe raw-comment union.

**Tech Stack:** Python 3.11, pytest, YAML configs, SQLite/vector-store frozen bundles, OpenAI-compatible vLLM, Markdown Skill instructions.

---

### Task 1: Specify probe isolation and Raw removal

**Files:**
- Modify: `tests/test_pipeline_mock.py`
- Modify: `tests/test_staged.py`
- Modify: `tests/test_panel_analysis.py`

- [ ] Add a recording LLM test with two probes and disjoint comment IDs.
- [ ] Assert Cue calls contain only their own probe's Top-10.
- [ ] Assert Summary and Personality issue one local call per probe, followed by one aggregation call whose input contains local outputs rather than raw comments.
- [ ] Assert `raw` is rejected as an unsupported formal condition.
- [ ] Update prepared-manifest and paired-analysis fixtures to the five retained conditions.

### Task 2: Implement native hierarchical condition processing

**Files:**
- Modify: `src/sideprofile/pipeline.py`
- Modify: `src/sideprofile/staged.py`

- [ ] Replace the six-condition constant with `none`, `personality`, `summary`, `gold`, and `ours`.
- [ ] Delete the full-union renderer and Raw payload branch.
- [ ] Render one probe's retrieved records at a time for Summary and Personality.
- [ ] Generate one local observation per probe and aggregate only those observations into the final condition payload.
- [ ] Preserve the existing `comments -> per-probe Cue -> Person Model` path for Ours.
- [ ] Keep every request free of numeric length instructions, truncation, API generation caps, retries, temperature, top-p, and seeds.

### Task 3: Align executable scope, analysis, and Skill

**Files:**
- Modify: `configs/scope.yaml`
- Modify: `configs/offline/panel-a.yaml`
- Modify: `configs/offline/panel-d.yaml`
- Modify: `configs/*.yaml`
- Modify: `scripts/build_experiment_audit.py`
- Modify: `scripts/analyze_panel_results.py`
- Modify: `skills/side-profile-experiment/SKILL.md`
- Modify: `skills/side-profile-experiment/references/operations.md`
- Modify: `skills/side-profile-experiment/references/analysis-report.md`
- Modify: `README.md`
- Modify: `plan.md`

- [ ] Remove Raw from every formal condition list, contrast list, report statement, and execution instruction.
- [ ] Define Summary and Personality as 24 isolated probe calls plus condition-native aggregation.
- [ ] Make preflight require the exact five-condition order.
- [ ] Keep the GPU Agent read-only with respect to corpus, retrieval, configs, and manifests.

### Task 4: Rebuild and verify the local frozen package

**Files:**
- Regenerate: `data/audits/experiment_inventory.json`
- Regenerate: `offline/code.manifest.json`
- Regenerate: `data/manifests/panel-a.json`
- Regenerate: `data/manifests/panel-d.json`

- [ ] Run the complete pytest suite.
- [ ] Run the experiment audit and Skill validator.
- [ ] Rebuild the code manifest and both panel manifests.
- [ ] Verify both panel bundles report `research_ready: true` with no failures.

### Task 5: Validate on smu_cluster before publishing

**Files:**
- Update installed Skill only after local validation.
- Update private GitHub repository only after remote validation.

- [ ] Re-read target rules and inventory the existing project, environment, model, output, and asset paths without exposing secrets.
- [ ] Copy the newly frozen package to a separate validation directory; do not edit frozen inputs on the GPU host.
- [ ] Run offline preflight, both experiment preflights, and the full pytest suite.
- [ ] Run real Qwen2.5-14B conditioning preparation far enough to complete all Panel A and Panel D characters and verify their immutable prepared directories.
- [ ] Exercise every registered Actor model against the new five-condition prepared format and run evaluator smoke checks without changing provider/vLLM defaults.
- [ ] Push only if all validation checks pass; otherwise preserve the failed artifacts and report the exact blocker without pushing.
