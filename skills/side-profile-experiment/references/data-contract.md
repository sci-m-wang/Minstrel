# Local-Only Character Comment Corpus Contract

This reference is for the local preparation side only. A downstream/GPU Agent must not execute any
instruction in this file, build or repair the corpus, or change data inputs. Its only valid response
to a missing or invalid corpus is to stop and return the failed preflight check.

Read this reference locally for data preparation, import, privacy, or coverage checks before freezing
the execution bundle.

## Model acquisition exception: Llama from ModelScope

The pinned `llama-3.1-8b-instruct` actor must be acquired on a connected preparation machine from
ModelScope, never from Hugging Face. `offline/models.yaml` fixes the ModelScope repository
`LLM-Research/Meta-Llama-3.1-8B-Instruct` at commit
`359efdbb8af05b788a4ad4185215c6b8caa9052c`. ModelScope CLI accepts the repository's `master`
revision rather than a raw commit. The preparation script therefore verifies that `master` resolves
to the pinned commit before invoking ModelScope and fails if it has moved. With the ModelScope CLI
available, run:

```bash
python3 scripts/download_offline_models.py \
  --asset-root "$SIDEPROFILE_ASSET_ROOT" \
  --only llama-3.1-8b-instruct
```

The registry excludes the duplicate `original/` checkpoint format and retains the official root
safetensors used by vLLM. The downloader does not set worker counts or retries. Build the model asset
manifest after download and let `scripts/offline_preflight.py` verify source, repository, revision,
file sizes, and hashes. The disconnected execution Agent never runs this command.

## Accepted inputs

Use UTF-8 JSONL or CSV. Each row must contain:

| Field | Requirement |
|---|---|
| `comment_id` | Stable unique ID; never reuse it for changed text. |
| `character_id` | Must exist in `data/catalog/characters.json`. |
| `character_name`, `work` | Private backend metadata; used only for audit and masking. |
| `platform` | Stable source label such as `reddit` or an authorized export name. |
| `thread_id` | Source thread ID when available. |
| `author_hash` | Salted, irreversible author identifier; do not store public usernames here. |
| `timestamp` | Source timestamp when available. |
| `raw_text` | Original comment text; minimum 12 characters. |
| `language` | BCP-47-like short tag such as `en` or `zh`. |
| `source_url` | Audit URL when retention is permitted. |
| `collection_method` | How the record was obtained, e.g. `authorized_export`. |
| `license_note` | Permission, terms, or dataset-license note. |
| `is_synthetic` | `false` for research evidence; synthetic fixtures must be `true`. |

Optional `collected_at` is filled automatically. Never import API tokens, email addresses, or raw
platform usernames into the corpus. Generate author hashes with a private per-study salt; do not
commit the salt.

Language and platform are not eligibility constraints. A Chinese comment may support an English-work
character and vice versa. Preserve the source language in `language`; a translation may be retained
as a derived audit artifact, but never replace or obscure the source text and URL. Target relevance is
the substantive gate: the text must provide evidence about the selected character rather than merely
appearing below a character-related page.

## Authorized acquisition only

The repository intentionally does not automate scraping PDB, Reddit, MyAnimeList, Douban, or Hupu.
Use an official API, licensed dataset, approved export, or user-provided file. Collection permission is
separate from analysis permission. Stop and ask for authority if neither the source terms nor the user
clearly authorize collection.

Browser-assisted acquisition must use normal visible navigation, respect access controls and site
terms, and hand login/CAPTCHA steps to the user. It remains a local preparation task and must never be
delegated to the GPU execution Agent.

## Import and audit

From the project root:

```bash
sideprofile init-corpus \
  --db data/corpus/comments.sqlite \
  --catalog data/catalog/characters.json

sideprofile import-comments \
  --db data/corpus/comments.sqlite \
  --catalog data/catalog/characters.json \
  --input <authorized-comments.jsonl>

sideprofile corpus-stats --db data/corpus/comments.sqlite
```

The import validates rows, enforces character existence, and deduplicates normalized text per
character. A duplicate count is not an error, but it must be recorded in the data log.

Before an English Panel A/B/C research run, require at least 500 comments, 2 platforms, and 100
independent authors per role unless the frozen design documents a different threshold:

```bash
sideprofile validate-corpus \
  --db data/corpus/comments.sqlite \
  --catalog data/catalog/characters.json \
  --min-comments 500 --min-platforms 2 --min-authors 100
```

For Panel D, use `--min-comments 500 --min-platforms 2 --min-authors 100`, exactly the same hard
coverage gate as Panel A. Do not pass `--include-synthetic` for research validation.

## Catalog maintenance

`character_id` is stable backend identity. `anonymous_id` is the only identity shown to profiling and
Actor calls. Add every known alias, translated name, surname, title, and work name needed for masking.
After changing aliases, rebuild profiles; cached artifacts created under the old mask are invalid.

Panel B's 15 role names must be imported from the official CharacterRAG manifest rather than guessed.
Panel/benchmark data must retain source version and license in the experiment log.

## Freeze handoff

After every coverage, source, privacy, gold-profile, and benchmark check passes, the preparation side
must build the persistent vector database before freezing. This command may run on a preparation GPU
machine controlled by the preparation side, but never by the downstream experiment Agent:

```bash
sideprofile build-vector-store \
  --db data/corpus/comments.sqlite \
  --output data/vector_store/qwen3-embedding-0.6b.sqlite \
  --model "$SIDEPROFILE_ASSET_ROOT/models/qwen3-embedding-0.6b" \
  --model-key qwen3-embedding-0.6b \
  --model-revision 97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3

sideprofile verify-vector-store \
  --db data/corpus/comments.sqlite \
  --vector-store data/vector_store/qwen3-embedding-0.6b.sqlite \
  --model-key qwen3-embedding-0.6b \
  --model-revision 97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3
```

The builder uses the model-native `query` prompt for the 24 English and 24 Chinese fixed queries,
encodes documents without a prompt, normalizes both sides, and stores float32 vectors for exact
cosine search. It deliberately does not create an approximate or quantized index. Build to a new
path; never overwrite a frozen database in place.

After vector verification passes, create the manifest referenced by the final config:

```bash
sideprofile freeze-bundle \
  --project-root . \
  --config configs/<frozen-run>.yaml \
  --output data/manifests/<panel>.json
sideprofile verify-bundle --project-root . --manifest data/manifests/<panel>.json
```

Only this verified bundle may be copied to the GPU host. Any later data/config change invalidates the
manifest and requires a new local freeze.
