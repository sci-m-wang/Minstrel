import hashlib
import json

from sideprofile.staged import _panel_key, verify_prepared


def test_verify_prepared_detects_checksum_changes(tmp_path) -> None:
    artifact = tmp_path / "characters" / "x" / "conditionings.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("{}\n", encoding="utf-8")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    manifest = {
        "stage": "conditioning_preparation",
        "research_valid": True,
        "profiler_provider": "GPT",
        "profiler_model": "gpt-5.6-sol",
        "retrieval_mode": "openai_exact_vector_recall+cohere_rerank",
        "embedding_provider": "GPT",
        "embedding_model": "text-embedding-3-small",
        "reranker_provider": "COHERE",
        "reranker_model": "Cohere-rerank-v4.0-pro",
        "candidate_top_k": 20,
        "final_top_k": 10,
        "conditions": ["none", "personality", "summary", "gold", "ours"],
        "comment_processing": "isolated_per_probe_top10",
        "condition_aggregation": {
            "personality": "per_probe_observations_then_aggregate",
            "summary": "per_probe_summaries_then_aggregate",
            "ours": "per_probe_cues_then_person_model",
        },
        "artifacts": [
            {"path": "characters/x/conditionings.json", "sha256": digest}
        ],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert verify_prepared(tmp_path)["ok"]
    artifact.write_text('{"changed":true}\n', encoding="utf-8")
    result = verify_prepared(tmp_path)
    assert not result["ok"]
    assert "checksum mismatch" in result["failures"][0]


def test_panel_key_maps_frozen_run_name_to_model_matrix() -> None:
    assert _panel_key({"name": "panel-a"}) == "panel_a"
    assert _panel_key({"name": "panel-d"}) == "panel_d"
