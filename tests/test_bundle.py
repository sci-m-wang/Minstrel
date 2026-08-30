from __future__ import annotations

import json
from pathlib import Path

import yaml

from sideprofile.bundle import build_bundle_manifest, verify_bundle_manifest
from sideprofile.corpus import CommentCorpus
from sideprofile.schema import CharacterSpec, Comment


def test_frozen_bundle_detects_changed_input(tmp_path: Path) -> None:
    (tmp_path / "configs").mkdir()
    (tmp_path / "data").mkdir()
    (tmp_path / "configs" / "scope.yaml").write_text("panels: [A]\n", encoding="utf-8")
    spec = CharacterSpec(
        character_id="x",
        character_name="Example",
        work="Example Work",
        anonymous_id="TARGET_X",
        panel="A",
        gold_profile="A careful person.",
    )
    catalog = tmp_path / "data" / "catalog.json"
    catalog.write_text(
        json.dumps({"characters": [spec.model_dump()]}, ensure_ascii=False), encoding="utf-8"
    )
    benchmark = tmp_path / "data" / "benchmark.jsonl"
    benchmark.write_text(
        json.dumps(
            {
                "example_id": "e1",
                "character_id": "x",
                "query": "A plan changes. What do you do?",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    db = tmp_path / "data" / "comments.sqlite"
    with CommentCorpus(db) as corpus:
        corpus.add_character(spec)
        corpus.add_comment(
            Comment(
                comment_id="c1",
                character_id="x",
                character_name="Example",
                work="Example Work",
                platform="forum",
                author_hash="author",
                raw_text="Example restores the plan carefully after unexpected changes.",
                language="en",
                license_note="test fixture",
            )
        )
    config = tmp_path / "configs" / "research.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "data": {
                    "corpus_db": "data/comments.sqlite",
                    "catalog": "data/catalog.json",
                    "benchmark": "data/benchmark.jsonl",
                    "include_synthetic": False,
                    "coverage": {
                        "min_comments": 1,
                        "min_platforms": 1,
                        "min_authors": 1,
                    },
                },
                "run": {
                    "name": "research-a",
                    "character_ids": ["x"],
                    "conditions": ["gold", "ours"],
                },
            }
        ),
        encoding="utf-8",
    )
    manifest = tmp_path / "data" / "manifest.json"
    build_bundle_manifest(
        project_root=tmp_path, config_path=config, output_path=manifest
    )
    assert verify_bundle_manifest(project_root=tmp_path, manifest_path=manifest)["ok"]
    benchmark.write_text(benchmark.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    result = verify_bundle_manifest(project_root=tmp_path, manifest_path=manifest)
    assert not result["ok"]
    assert "checksum mismatch" in result["failures"][0]
