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
        "conditions": ["none", "personality", "raw", "summary", "gold", "ours"],
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
