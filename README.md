# SideProfile

SideProfile implements the experiment in `plan.md`: build an auditable character-comment
corpus, retrieve evidence for 24 fixed profiling probes, extract evidence-cited cues, construct
an identity-blind person model, run role responses, and produce a result report.

Evidence retrieval is completed on the connected preparation side: a persistent exact-cosine vector
database built through `text-embedding-3-small` stores every non-synthetic comment and all 24 probes
in English and Chinese; each probe recalls Top-20 candidates, `Cohere-rerank-v4.0-pro` scores all 20,
and local code retains Top-10. BM25 and RRF are not part of the research method. GPU Actor runs use
only the frozen prepared payloads and never call either retrieval service.

The bundled smoke corpus is explicitly synthetic and exists only to test wiring. It must never
be used as experimental evidence. Research comments come from documented public pages, official
APIs, and browser-visible discussions; raw text remains private and source/relevance audits are
preserved in the frozen bundle.

The formal corpus applies two checksum-audited deterministic filters: Stack Exchange contributes
only `#comment` records (not question or answer bodies), and every non-synthetic record with at
least 1,000 Unicode characters is excluded. Per-role comment counts are reported as descriptive
sample sizes rather than forced to a common minimum.

The responsibility boundary is strict: this local workspace performs source validation, browser/API
collection, relevance adjudication, author hashing, corpus/benchmark construction, tests, and bundle
freezing. A downstream GPU Agent receives the complete frozen repository and corpus plus pinned model
assets and an offline wheelhouse. It must not collect, import, synthesize, translate, repair, or modify
any experimental input. See
`data/sources/feasibility.md` for the current platform-access audit.

The executable scope is intentionally limited to Panels A and D with five conditions: `none`,
`personality`, `summary`, `gold`, and `ours`. Summary, Personality, and Cue extraction process each
probe's Top-10 independently before their condition-native aggregation. External AMADEUS, RoleGPT/RoleLLM,
PersonaForge, and CoSER baselines are excluded until their official code, inputs, and model artifacts
can be pinned and tested. See `configs/scope.yaml`.

The frozen actor matrix contains only models whose pinned weights are present in the offline asset
bundle. Llama-3.1-8B-Instruct is sourced specifically from the pinned ModelScope mirror rather than
Hugging Face. It is not replaced or approximated, and the five internal conditions are frozen.

The connected preparation side also uses the fixed `GPT/gpt-5.6-sol` Profiler to build and freeze both
five-condition prepared directories. The embedding, reranking, and profiling `.env` is never shipped. For GPU deployment, two
preflights, verification and static token auditing of the supplied prepared directories, all Actor
runs, official evaluation, panel-level paired analysis, and the final report, use the
agent-executable `skills/side-profile-experiment/SKILL.md`.

## Installation and frozen assets

This private repository contains the experiment code, tests, configs, official benchmark/profile
inputs, frozen research corpus, corpus inventory metadata, the frozen `text-embedding-3-small`
vector database, Cohere rerank provenance, model
registry, and execution Skill. Keep repository access restricted because the corpus contains public
comments retained for private research use whose source terms do not authorize redistribution. The
repository deliberately excludes model weights, environments, API secrets, author salt, private
collection traces, and run outputs. All runtime data are checksum-verified by the frozen manifests.

See [`docs/INSTALL.md`](docs/INSTALL.md) for the connected preparation workflow, pinned model
download rules, Linux/Python/CUDA environment, offline wheelhouse construction, restricted-data
placement, target-machine discovery, preflight, and execution entry points. See
[`data/README.md`](data/README.md) for the repository data boundary.
