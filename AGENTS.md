# SideProfile agent instructions

Before running experiments or analyzing results on a compute/GPU host, read and follow
`skills/side-profile-experiment/SKILL.md` and only the references it routes to. The GPU-side Agent may
only verify and execute a locally frozen bundle. It must never collect/import comments, build/repair
SQLite, edit catalog or benchmark data, create any profile/conditioning, read a GPT `.env`, relax
coverage, or regenerate configs and manifests. The supplied five-condition prepared directories were
built locally with the fixed `GPT/gpt-5.6-sol` Profiler and are immutable GPU-side inputs. A data-side or
prepared-side preflight failure must be returned to the local preparation side unchanged.

The hard-gated core scope is `configs/scope.yaml`: Panels A/D with `none`, `personality`, `summary`,
`gold`, and `ours`. Raw is not executable. After core preflight, attempt each available official
AMADEUS, RoleGPT/RoleLLM, PersonaForge, and CoSER artifact in isolation. A baseline failure is recorded
and does not block or invalidate the core; never approximate or modify a baseline. Do not set API output-token caps,
call budgets, retry limits, `temperature`, `top_p`, or API seeds. Run the Skill preflight before every
configured experiment and report the frozen bundle hash status, immutable run directory, and
`research_valid` status.
