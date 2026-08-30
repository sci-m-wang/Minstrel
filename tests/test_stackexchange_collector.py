import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "collect_stackexchange.py"
SPEC = importlib.util.spec_from_file_location("stackexchange_collector", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


def test_plain_text_removes_markup_and_code() -> None:
    assert MODULE.plain_text("<p>Hermione &amp; Ron</p><pre><code>x()</code></pre>") == "Hermione & Ron"


def test_capture_row_preserves_provenance() -> None:
    row = MODULE.capture_row(
        kind="answer",
        item={"answer_id": 7, "body": "<p>A sufficiently long observation.</p>", "owner": {"user_id": 9}},
        question_id=3,
        source_title="Title",
        source_url="https://example.test/a/7",
    )
    assert row["source_comment_id"] == "answer:7"
    assert row["author_source_id"] == "9"
    assert row["target_relevant"] is None
