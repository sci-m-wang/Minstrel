# Best-effort External Baselines

Read this reference only when attempting AMADEUS, RoleGPT/RoleLLM, PersonaForge, or CoSER. These are
external comparison methods, not Ours conditions. Their availability is desirable for comparison but
is not a hard gate for the frozen Panel A/D core experiment.

## Priority and isolation

Pass the core bundle, prepared-conditioning, and Actor-input preflights first. Reserve the run
directories and GPUs needed for Ours. Use only remaining capacity for a concurrent baseline, or wait
until the core finishes. Run each baseline in its own official checkout, environment, model directory,
and output directory. A baseline setup or runtime failure must not delay or change a core config,
prepared directory, manifest, result, or `research_valid` value.

Attempt these planned comparisons:

| Method | Planned panel | Required official basis |
| --- | --- | --- |
| AMADEUS | B / CharacterRAG | Author-released CharacterRAG/AMADEUS code, CharacterRAG documents and 450 QA |
| RoleGPT / RoleLLM | C / RoleBench | `InteractiveNLP-Team/RoleLLM-public`, native RoleBench data, profiles and checkpoints |
| PersonaForge | C / RoleBench | Author-released PersonaForge code and artifacts linked from the ACL paper |
| CoSER | C / RoleBench | Author-released CoSER code, specialized checkpoint and native evaluation assets |

`configs/baselines-smoke.yaml` is an internal synthetic pipeline smoke test. It is not any external
baseline and must never be used as evidence that one of the methods above ran.

## Attempt protocol

For each method:

1. Inspect the target's supplied baseline roots and model/data inventory. Do not assume paths from a
   previous machine and do not use the network when the target rules forbid it.
2. Locate an official checkout or release. Record its remote/source, exact git commit or release
   revision, license, and hashes of supplied data/checkpoints. If any required artifact is absent,
   write an `unavailable` status and continue to the next method.
3. Read that revision's official README and run its documented environment check or smallest native
   smoke command. Use the official dependencies and native method unchanged. Do not inject
   `max_tokens`, retry policies, call budgets, decoding controls, prompt rewrites, replacement models,
   substitute profiles, or substitute datasets. Do not edit official code to fit the machine.
4. If the native smoke succeeds and the planned Panel B/C inputs exist, run the official evaluation.
   Preserve stdout, stderr, command, environment/package inventory, outputs, and official metric.
5. If a command fails, preserve the failure evidence, mark the method `failed`, and continue. Do not
   repair it by changing the method. Follow-up diagnosis is allowed only as read-only inspection.

## Required attempt record

Write one file at `runs/external-baselines/<method>/attempt-status.json` with at least:

```json
{
  "method": "RoleLLM",
  "status": "completed | failed | unavailable",
  "official_source": "source URL or release identifier",
  "revision": "exact commit/release or null",
  "modified_official_code": false,
  "dataset": "native dataset/revision or null",
  "checkpoint": "native checkpoint/revision or null",
  "commands": [],
  "exit_codes": [],
  "outputs": [],
  "official_metrics": {},
  "failure_stage": null,
  "failure_reason": null,
  "eligible_for_comparison": false
}
```

Set `eligible_for_comparison=true` only when the official pipeline completes on the planned native
data with pinned provenance and its official evaluator. Do not convert a failed/missing method to a
zero score, do not compare incompatible native metrics as if paired, and do not describe an attempt
as a reproduced baseline unless this gate passes.
