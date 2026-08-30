# SideProfile

SideProfile implements the experiment in `plan.md`: build an auditable character-comment
corpus, retrieve evidence for 24 fixed profiling probes, extract evidence-cited cues, construct
an identity-blind person model, run role responses, and produce a result report.

Dense evidence retrieval uses a preparation-side, persistent exact-cosine vector database built with
the pinned `Qwen/Qwen3-Embedding-0.6B` revision. It stores embeddings for every non-synthetic comment
and all 24 probes in English and Chinese. Formal runs only read this frozen database; they do not
download an embedding model or rebuild vectors.

The bundled smoke corpus is explicitly synthetic and exists only to test wiring. It must never
be used as experimental evidence. Research comments come from documented public pages, official
APIs, and browser-visible discussions; raw text remains private and source/relevance audits are
preserved in the frozen bundle.

The responsibility boundary is strict: this local workspace performs source validation, browser/API
collection, relevance adjudication, author hashing, corpus/benchmark construction, tests, and bundle
freezing. A downstream GPU Agent receives the complete frozen repository and corpus plus pinned model
assets and an offline wheelhouse. It must not collect, import, synthesize, translate, repair, or modify
any experimental input. See
`data/sources/feasibility.md` for the current platform-access audit.

The executable scope is intentionally limited to Panels A and D with six conditions: `none`,
`personality`, `raw`, `summary`, `gold`, and `ours`. External AMADEUS, RoleGPT/RoleLLM,
PersonaForge, and CoSER baselines are excluded until their official code, inputs, and model artifacts
can be pinned and tested. See `configs/scope.yaml`.

The frozen actor matrix contains only models whose pinned weights are present in the offline asset
bundle. Llama-3.1-8B-Instruct is sourced specifically from the pinned ModelScope mirror rather than
Hugging Face. It is not replaced or approximated, and the six controlled conditions are unchanged.

The connected preparation-side commands for corpus import and bundle freezing are intentionally not
part of the GPU procedure. For deployment, two preflights, shared conditioning preparation, all actor
runs, official evaluation, panel-level paired analysis, and the final report, use the agent-executable
`skills/side-profile-experiment/SKILL.md`.

## Installation and frozen assets

The public repository contains the experiment code, tests, configs, official benchmark/profile
inputs, corpus inventory metadata, the frozen Qwen3 vector database, model registry, and execution
Skill. It deliberately does not contain model weights, environments, API secrets, or comment text
whose source terms do not authorize redistribution. The research corpus is supplied separately to
authorized experiment hosts and is checksum-verified by the frozen manifests.

See [`docs/INSTALL.md`](docs/INSTALL.md) for the connected preparation workflow, pinned model
download rules, Linux/Python/CUDA environment, offline wheelhouse construction, restricted-data
placement, target-machine discovery, preflight, and execution entry points. See
[`data/README.md`](data/README.md) for the public/restricted data boundary.
