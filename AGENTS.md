# SideProfile agent instructions

Before running experiments or analyzing results on a compute/GPU host, read and follow
`skills/side-profile-experiment/SKILL.md` and only the references it routes to. The GPU-side Agent may
only verify and execute a locally frozen bundle. It must never collect/import comments, build/repair
SQLite, edit catalog or benchmark data, create gold profiles, relax coverage, or regenerate configs
and manifests. A data-side preflight failure must be returned to the local preparation side unchanged.

The executable scope is `configs/scope.yaml`: Panels A/D with `none`, `personality`, `raw`, `summary`,
`gold`, and `ours`. Do not approximate removed external baselines. Do not set API output-token caps,
call budgets, retry limits, `temperature`, `top_p`, or API seeds. Run the Skill preflight before every
configured experiment and report the frozen bundle hash status, immutable run directory, and
`research_valid` status.
