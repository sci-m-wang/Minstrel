# Complete Research Result Delivery

The execution-side Agent owns delivery, not just computation. Push the detailed results to
`https://github.com/sci-m-wang/Minstrel` after verifying that the destination is private. Apply this
contract to every configured panel, Actor, condition and replicate, including completed, failed and
partial runs and the available external-baseline artifacts. Do not deliver only selected examples
or a lightweight review ZIP.

## Preserve originals and establish what exists

For a new run, retain each complete Actor request and raw response, usage, finish reason and errors
in the run trace as execution proceeds. Keep the full generated text before any evaluator-specific
processing, including first-line selection. Preserve scoring inputs and per-example scores separately.
Do not change prompts, decoding, experimental conditions, evaluator behavior, or frozen inputs merely
to package outputs. If the supplied runtime cannot retain required evidence, report that limitation
before model calls; request any needed code change from the preparation side.

For an existing run, inventory the original directories and stored snapshots before claiming that
files were saved. Open the snapshot and verify its contents; a path, upload receipt or object size is
not evidence that raw answers and traces are actually present. Recover from an available durable
copy if the execution machine has been cleared. Do not rerun generation to replace lost originals,
expand summaries into fabricated records, or label reconstructed requests as actual request traces.
Deliver available evidence and list missing files explicitly.

Before releasing ephemeral storage, ensure the complete artifact set has a verified durable copy.
An approved object-store snapshot may be an intermediate backup, but it does not replace the final
repository delivery. If network, permission or storage policy blocks transfer, preserve what is
available and report delivery as blocked or partial rather than complete. Never bypass site rules.

## Required artifact set

Keep the original run IDs and record-level identifiers: panel, Actor/model, character, example,
condition, replicate and prepared-manifest hash. The reviewer must be able to join answers, exact
requests, scores and provenance without guessing.

| Category | What must be committed and pushed |
| --- | --- |
| Actor originals | Complete `generations.jsonl`, `trace.jsonl` and any separate raw response files; full system/user messages, benchmark context and question, full multiline output, available usage/finish reasons/errors. Preserve all conditions and replicates, not only scored or successful answers. |
| Official evaluation | Full per-example Reaction outputs, CharacterRM scoring inputs and `charactereval-evaluation.json`, other official evaluator artifacts, pending General Response answers, and applicable baseline native outputs. Keep full raw text distinct from text actually sent to an evaluator. |
| Analysis | All `condition-summary.csv`, `paired-contrasts.csv`, `analysis.json`, reports, tables, figures and analysis scripts actually used; aggregate files do not substitute for per-example records. |
| Runtime evidence | Per-run manifests/status, frozen-input and prepared hash receipts, input/context audits, deployment and package-version inventories, launch commands, stdout/stderr, failure/recovery logs and the baseline attempt ledger. Record unavailable methods as unavailable, not as successful runs. |
| Exact executed source | Source, prompts, conversion/scoring/analysis scripts, launch configuration, dependency specifications and code manifest that actually produced the outputs, with original revision and any uncommitted patch/untracked execution files. A list of source hashes alone is insufficient. |
| Prepared provenance | Original prepared manifests plus Cue/profile, conditioning and retrieval artifact paths and hashes. If identical bytes already exist in Minstrel, cite their exact commit and paths and verify those bytes; do not rebuild or needlessly duplicate them. Record the data/model/tokenizer revisions without uploading model weights. |

Store the executed version separately from newer repository code. Do not silently replace it with a
cleaned-up implementation, overwrite the original run manifests, or regenerate the frozen input/code
manifests to make historical execution appear to use the latest Skill. Delivery-only instructions
can be read in a separate checkout; they do not change the run's recorded execution revision.

## Private-repository publication

1. Verify the repository owner/name and current visibility immediately before staging a delivery.
   Private status from an earlier task is not sufficient. If it is public or cannot be confirmed,
   do not upload research artifacts; report the issue and request explicit user approval to make it
   private. Keep delivery blocked until private status can be verified.
   Do not automatically change visibility or upload first with a promise to make it private later.
2. Use a separate delivery checkout and a new result branch. Preserve unrelated changes and existing
   frozen artifacts. A suitable layout is `deliveries/<delivery-id>/runs/<original-run-id>/`,
   `analysis/`, `control/` and `executed-source/`. Record any path mapping. A reusable source commit
   is sufficient only if all actually executed files are retrievable there; otherwise include a
   source snapshot and patch. Do not force-push or rewrite existing history.
3. Exclude model/checkpoint weights, environments, wheelhouses, caches, `.env` files, credentials,
   authentication headers and author salts. Review staged paths and content without printing secrets.
   If an original trace contains a secret, retain the unmodified original securely outside Git,
   redact only that secret in an explicitly labeled delivery copy, and document the redaction without
   revealing its value. Do not use privacy review to shorten scientific inputs or answers.
4. The project ignores `runs/`, `results/`, `reports/` and `*.log`. Do not mistake an ignored file for
   an uploaded file. Stage only the reviewed delivery paths (including explicitly approved ignored
   logs); never use a broad force-add of the whole workspace or relax global secret exclusions.
5. Use Git LFS for files exceeding ordinary GitHub blob limits and push the actual LFS objects, not
   just pointer files. Do not omit or truncate large traces. If LFS is unavailable, a lossless archive
   split within repository limits is acceptable when accompanied by part hashes and exact reassembly
   instructions; verify that the reassembled files match their originals. An external-storage link
   alone is not an alternative to the requested repository upload. Do not purchase storage or change
   account plans without approval; disclose any blocker and preserve the durable originals.
6. Commit and push the reviewed delivery branch using the approved connected host. A disconnected
   GPU node must hand off the full artifacts through its authorized transfer route to a connected
   host that performs the push; do not ask it to bypass the offline policy. Do not fetch new models,
   call external inference APIs or regenerate data during delivery.

## Verify the remote copy and report completion

Create an output-side artifact index containing relative path, byte count and SHA-256 for every
delivered artifact, its original location, counts by run/condition/replicate, and missing items. Mark
redacted files and identify the original executed revision, code-manifest hash, input-bundle hashes
and prepared-manifest hashes. The index does not hash itself. Keep it separate from immutable
experiment manifests.

After pushing, verify the remote branch commit and retrieve the delivery from a fresh checkout with
LFS content materialized or archives reassembled. Compare the delivered file hashes with the index
and verify that JSON/JSONL can be read and answer/score counts agree with the recorded run grid.
Check all available originals, not just the summary report. Do not hide failed or unscored records.
If any verification fails, report the specific incomplete delivery without relabeling an otherwise
valid experiment as invalid solely because transfer failed.

Return the repository URL, branch, pushed commit SHA, artifact-index path, executed-source path,
remote verification result, missing-original-file list, computation status and `research_valid`.
Declare delivery complete only when the required detailed artifacts are retrievable and verified in
the private repository. A local commit, summary ZIP, S3 pointer or successful upload without readback
does not satisfy this finish criterion.
