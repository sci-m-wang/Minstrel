# Private data availability

This repository must remain private. It includes:

- the frozen research comment corpus at `data/corpus/comments.sqlite`;
- Panel A and Panel D official benchmark/profile inputs with pinned upstream revisions;
- character catalogs and identity-masking aliases;
- source feasibility, corpus coverage, import, experiment, and bundle inventories;
- relevance decisions and rationales that do not reproduce source comment text;
- the frozen `text-embedding-3-small` exact-cosine vector database and Cohere rerank provenance;
- synthetic smoke fixtures.

The repository excludes:

- `data/private/`, which contains author salt and private collection/classification traces;
- API credentials, browser state, model weights, environments, and run outputs.

Most corpus source terms allow private research retention but not public redistribution. Do not make
this repository public or copy the corpus outside authorized experiment hosts. The bundle manifests
verify its checksum. The GPU Agent must return a missing or mismatched corpus failure unchanged; it
must never collect, rebuild, edit, translate, or substitute research comments.
