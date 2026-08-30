import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "collect_bilibili.py"
SPEC = importlib.util.spec_from_file_location("bilibili_collector", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


def test_capture_row_preserves_public_api_provenance() -> None:
    row = MODULE.capture_row(
        item={
            "rpid": 123,
            "ctime": 1_700_000_000,
            "content": {"message": "华妃在失去信任时会用强势行为保护自己的地位。"},
            "member": {"mid": 456, "uname": "public-user"},
        },
        bvid="BV1test",
        aid=789,
        title="华妃人物分析",
    )
    assert row is not None
    assert row["source_comment_id"] == "123"
    assert row["author_source_id"] == "456"
    assert row["collection_method"] == "bilibili_public_comment_api"
    assert row["target_relevant"] is None
    assert row["source_url"].endswith("#reply123")
