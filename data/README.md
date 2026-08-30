# Data availability

The public repository includes:

- Panel A and Panel D official benchmark/profile inputs with pinned upstream revisions;
- character catalogs and identity-masking aliases;
- source feasibility, corpus coverage, import, experiment, and bundle inventories;
- relevance decisions and rationales that do not reproduce source comment text;
- the frozen Qwen3-Embedding-0.6B exact-cosine vector database;
- synthetic smoke fixtures.

The public repository excludes:

- `data/corpus/comments.sqlite`, because most source comment terms allow private research retention
  but not public redistribution;
- `data/private/`, which contains author salt and private collection/classification traces;
- API credentials, browser state, model weights, environments, and run outputs.

An authorized formal execution receives the exact frozen corpus separately and places it at
`data/corpus/comments.sqlite`. The bundle manifests verify its checksum. The GPU Agent must return a
missing or mismatched corpus failure unchanged; it must never collect, rebuild, edit, translate, or
substitute research comments.
