# Analysis and Result Report

Read this reference when analyzing completed runs or preparing a shareable result report.

## Eligibility gate

Open `manifest.json` first. A run is not eligible for research claims if any of the following holds:

- `research_valid` is false or `include_synthetic` is true;
- the requested role lacks the documented corpus thresholds;
- the prepared retrieval mode, service identities, Top-20 candidate sets, Cohere scores, or Top-10
  outputs differ from the frozen `vector_rerank` contract;
- Actor/profiling/judge model versions or decoding differ across controlled conditions;
- official benchmark scoring is missing for a claimed benchmark result;
- identity leakage or invalid Cue citations are unresolved.

Retain ineligible runs as engineering evidence only and say why.

## Rebuild the base report

```bash
sideprofile analyze --run-dir <immutable-run-directory>
```

This creates `summary.csv` and `report.md`. Validate that response/scored counts equal the frozen grid.
Treat missing cells as missing, never as zero.

## Required analysis

For each panel, report condition mean, standard deviation or confidence interval, sample count, roles,
Actor, evaluator, replicate count, and corpus coverage. Internal condition analyses are:

- Ours vs Generic Summary: value beyond summarization;
- Ours vs Personality Only: person model beyond traits;
- Ours vs Anonymous Gold Profile: recovered fraction of oracle persona information.

Use paired comparisons on the same character/query/replicate units. Prefer a paired bootstrap confidence
interval over an unpaired test. Report absolute differences and uncertainty, not only p-values. With
multiple metrics or ablations, identify the correction method or label exploratory analyses.

Panel A addresses feasibility and reconstruction. Panel D tests Chinese and subtle-character
generalization. Panels B/C and their AMADEUS, RoleGPT/RoleLLM, PersonaForge, and CoSER comparisons
are outside the current executable scope and must not appear in result claims. Analyze iconicity and
identity recoverability as planned, without redefining groups after inspecting scores.

## Evidence audit

Sample Cue files across high, median, and low scoring roles. Confirm that citations occur in
`retrieval.json`, support comes from independent observers where status is `SUPPORTED`,
counterevidence is preserved, and unknown claims are not silently promoted. Include representative
success and failure cases without exposing author identities.

## Report structure

Produce a self-contained Markdown report with:

1. Executive finding and validity status.
2. Frozen design: panel, characters, models, conditions, replicates, evaluators, each condition's
   native information source, and the complete selected comment-ID sets used by comment-derived
   conditions. State explicitly that lengths are neither targeted nor normalized.
3. Corpus audit: comments, platforms, authors, language, duplicates, and excluded rows per character.
4. Primary result tables with paired uncertainty.
5. Ablations: coverage, breadth/depth, and identity leakage only when actually run.
6. Error and evidence analysis, including contested and unknown cues.
7. Cost: LLM calls/tokens, wall time, failures, and retries.
8. Context audit: maximum observed prompt/completion/total tokens and minimum remaining native
   context for per-probe Cue, Summary, Personality, aggregation, Person Model, and Actor calls.
9. Limitations: platform selection bias, social-perception bias, model judge limits, licensing, and
   benchmark contamination risk.
10. Reproduction commands and immutable artifact paths.

Never describe a smoke score as model quality, claim causality from an uncontrolled comparison, or
replace a missing official score with the built-in GPT judge without an explicit auxiliary label.
