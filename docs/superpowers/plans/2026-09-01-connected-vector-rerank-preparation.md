# Connected Vector Retrieval and Reranking Implementation Plan

> **For agentic workers:** Execute this plan inline in the current task. Do not delegate profile construction or move any connected-preparation responsibility to the GPU host.

**Goal:** Replace the research retrieval path with locally prepared `text-embedding-3-small` exact-vector recall followed by `Cohere-rerank-v4.0-pro`, then freeze all retrieval and five-condition profile artifacts for read-only GPU Actor execution.

**Architecture:** The connected preparation machine embeds the 8,673 frozen comments and all 24 bilingual probes through the configured OpenAI-compatible embedding endpoint, stores exact cosine vectors in a checksum-covered SQLite database, recalls the configured Top-20 candidates per character/probe, and sends those candidates to the configured Cohere v2 rerank endpoint. Local preparation keeps the returned ordering, takes Top-10, and gives each probe's isolated Top-10 to GPT-5.6 Sol profile construction. The GPU host verifies the frozen vector store, retrieval records, profiles, configs, and manifests but never loads API credentials or reruns retrieval/profile preparation.

**Tech Stack:** Python 3.11+, SQLite, NumPy exact cosine search, OpenAI-compatible embeddings HTTP API, Cohere v2 rerank HTTP API, GPT-5.6 Sol Profiler, pytest.

---

### Task 1: Lock the connected-service contracts

**Files:**
- Create: `src/sideprofile/connected_retrieval.py`
- Test: `tests/test_connected_retrieval.py`

- [ ] Define environment-backed embedding and reranker settings without exposing secrets.
- [ ] Implement one-request JSON clients using verified TLS and provider defaults; do not add application retries or token/document truncation parameters.
- [ ] Require one embedding per input and one finite Cohere score per submitted candidate.
- [ ] Record model, input hashes, result indices/scores, usage metadata, and timestamps in preparation-side traces.

### Task 2: Build and verify the OpenAI vector store

**Files:**
- Modify: `src/sideprofile/vector_store.py`
- Modify: `src/sideprofile/cli.py`
- Test: `tests/test_vector_store.py`

- [ ] Replace local Sentence Transformers construction with connected `text-embedding-3-small` construction.
- [ ] Preserve the exact-cosine SQLite schema, corpus/text fingerprints, bilingual probe hashes, and non-overwrite build behavior.
- [ ] Record the exact requested/returned model and aggregate API usage; transport batching must not alter vectors or retrieval results.
- [ ] Make verification accept a provider-managed API model without inventing a local revision.

### Task 3: Replace hybrid retrieval with vector recall and Cohere reranking

**Files:**
- Modify: `src/sideprofile/retrieval.py`
- Modify: `src/sideprofile/pipeline.py`
- Modify: `src/sideprofile/staged.py`
- Test: `tests/test_retrieval.py`
- Test: `tests/test_staged_profiler_contract.py`

- [ ] Remove BM25, RRF, Qwen3, and local CrossEncoder from the research execution path.
- [ ] Recall exact-vector Top-20, rerank all 20 through Cohere, and locally select Top-10.
- [ ] Preserve anonymization, deterministic tie-breaking, comment IDs, and isolated per-probe processing.
- [ ] Freeze embedding/reranker providers and models plus the rerank trace in the prepared manifest.

### Task 4: Update configs, preflight, documentation, and Skill

**Files:**
- Modify: `configs/offline/panel-a.yaml`
- Modify: `configs/offline/panel-d.yaml`
- Modify: `configs/research.example.yaml`
- Modify: `configs/panel-d.example.yaml`
- Modify: `offline/models.yaml`
- Modify: `scripts/build_experiment_audit.py`
- Modify: `skills/side-profile-experiment/SKILL.md`
- Modify: `skills/side-profile-experiment/references/operations.md`
- Modify: `skills/side-profile-experiment/references/data-contract.md`
- Modify: `skills/side-profile-experiment/scripts/preflight.py`
- Modify: `plan.md`
- Modify: `docs/INSTALL.md`

- [ ] Declare `vector_rerank`, `text-embedding-3-small`, `Cohere-rerank-v4.0-pro`, candidate Top-20, and final Top-10 consistently.
- [ ] Remove local Qwen3/BGE asset and runtime requirements.
- [ ] Keep credentials and all connected calls on the local preparation side only.
- [ ] GPU preflight must checksum/verify frozen inputs and prepared metadata without requiring OpenAI/Cohere connectivity.

### Task 5: Validate and freeze the local preparation bundle

**Files:**
- Replace after verification: `data/vector_store/text-embedding-3-small.sqlite`
- Create: `data/audits/connected_retrieval_services.json`
- Regenerate: `data/audits/experiment_inventory.json`
- Regenerate: `offline/code.manifest.json`
- Regenerate: `data/manifests/panel-a.json`
- Regenerate: `data/manifests/panel-d.json`
- Update: `skills/side-profile-experiment/`

- [ ] Run focused unit tests and the full test suite.
- [ ] Build the full 8,673-comment vector store and verify every comment/probe hash.
- [ ] Prepare Panel A and Panel D five-condition directories with GPT-5.6 Sol and Cohere traces.
- [ ] Verify all prepared artifacts and run static Actor-input audits before publishing.
- [ ] Run the Skill validator and regenerate all checksum authorities only after code/data stop changing.
