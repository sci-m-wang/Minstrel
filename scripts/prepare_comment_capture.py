#!/usr/bin/env python3
"""Validate a temporary browser capture and import only target-relevant comments.

The capture file is a local, short-lived JSONL artifact. It may contain a platform author ID but must
not contain passwords, cookies, email addresses, or browser/session data. The final corpus stores only
a salted hash of that author ID.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from sideprofile.corpus import CommentCorpus, hash_author, load_character_catalog
from sideprofile.llm import LLMClient, ProviderSettings
from sideprofile.schema import Comment


class CaptureRow(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    platform: str
    thread_id: str
    source_comment_id: str
    author_source_id: str
    raw_text: str
    source_url: str
    source_title: str
    timestamp: str = ""
    language: str = ""
    collection_method: str = "browser_visible_export"
    license_note: str
    target_relevant: bool | None = None

    @field_validator("raw_text")
    @classmethod
    def nonempty_text(cls, value: str) -> str:
        value = " ".join(value.split())
        if not value:
            raise ValueError("empty comment text")
        return value


def read_capture(path: Path) -> list[CaptureRow]:
    rows: list[CaptureRow] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(CaptureRow.model_validate_json(line))
            except Exception as exc:  # noqa: BLE001
                raise ValueError(f"invalid capture row at {path}:{line_no}: {exc}") from exc
    return rows


def ensure_private_salt(path: Path) -> str:
    if path.is_file():
        return path.read_text(encoding="utf-8").strip()
    path.parent.mkdir(parents=True, exist_ok=True)
    salt = secrets.token_urlsafe(48)
    path.write_text(salt + "\n", encoding="utf-8")
    os.chmod(path, 0o600)
    return salt


def detect_language(text: str) -> str:
    cjk = sum("\u3400" <= char <= "\u9fff" for char in text)
    letters = sum(char.isalpha() for char in text)
    return "zh" if cjk and cjk >= max(2, letters // 5) else "en"


def redact_public_usernames(text: str) -> str:
    # Remove platform @handles while preserving the substantive comment.
    text = re.sub(r"(?<!\w)@[A-Za-z0-9_][A-Za-z0-9_.-]{1,63}", "[USER]", text)
    text = re.sub(
        r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
        "[EMAIL]",
        text,
        flags=re.IGNORECASE,
    )
    return " ".join(text.split())


def stable_comment_id(row: CaptureRow, character_id: str) -> str:
    payload = "\0".join(
        [row.platform, row.thread_id, row.source_comment_id, row.source_url]
    ).encode("utf-8")
    source_id = f"{row.platform}_{hashlib.sha256(payload).hexdigest()[:20]}"
    # One public comment may independently contain usable evidence about multiple characters.
    # Keep source-level deduplication within a character without suppressing those valid many-to-many
    # associations across character corpora.
    return f"{character_id}:{source_id}"


def classify_relevance(
    llm: LLMClient, *, character_name: str, aliases: list[str], work: str, row: CaptureRow
) -> dict[str, Any]:
    return llm.chat_json(
        system=(
            "Decide whether one public third-party comment supplies evidence about the target "
            "fictional character. Relevant evidence includes behavior, motives, values, appraisals, "
            "affect/coping, relationships, self-narrative, situation-response patterns, or expressive "
            "style. Reject comments only about the work, actor, uploader, fandom, another character, "
            "or the commenter's own life unless they also make a substantive claim about the target. "
            "The comment may use pronouns or translated names. Return JSON with relevant (boolean), "
            "evidence_scope (short string), and rationale (one sentence)."
        ),
        user=json.dumps(
            {
                "target": character_name,
                "aliases": aliases,
                "work": work,
                "source_title": row.source_title,
                "comment": row.raw_text,
            },
            ensure_ascii=False,
        ),
        agent="comment_relevance",
    )


def classify_relevance_batch(
    llm: LLMClient, *, character_name: str, aliases: list[str], work: str, rows: list[CaptureRow]
) -> dict[str, dict[str, Any]]:
    payload = llm.chat_json(
        system=(
            "Independently decide whether each public third-party comment supplies evidence about "
            "the target fictional character. Relevant evidence includes behavior, motives, values, "
            "appraisals, affect/coping, relationships, self-narrative, situation-response patterns, "
            "or expressive style. Reject comments only about the work, actor, uploader, fandom, "
            "another character, or the commenter's own life unless they also make a substantive "
            "claim about the target. Pronouns and translated names are valid. Return JSON with a "
            "decisions array. Every item must contain source_comment_id, relevant (boolean), "
            "evidence_scope (short string), and rationale (one sentence). Judge each item on its own."
        ),
        user=json.dumps(
            {
                "target": character_name,
                "aliases": aliases,
                "work": work,
                "items": [
                    {
                        "source_comment_id": row.source_comment_id,
                        "source_title": row.source_title,
                        "comment": row.raw_text,
                    }
                    for row in rows
                ],
            },
            ensure_ascii=False,
        ),
        agent="comment_relevance_batch",
    )
    decisions = payload.get("decisions")
    if not isinstance(decisions, list):
        raise ValueError("batch relevance response lacks decisions array")
    by_id = {
        str(item.get("source_comment_id")): item
        for item in decisions
        if isinstance(item, dict)
    }
    expected = {row.source_comment_id for row in rows}
    if set(by_id) != expected:
        raise ValueError("batch relevance response does not cover the exact requested IDs")
    return by_id


def prepare_rows(
    rows: list[CaptureRow],
    *,
    character: Any,
    salt: str,
    llm: LLMClient | None,
    decisions_path: Path | None = None,
) -> tuple[list[Comment], list[dict[str, Any]]]:
    comments: list[Comment] = []
    decisions: list[dict[str, Any]] = []
    existing: dict[tuple[str, str, str], dict[str, Any]] = {}
    decision_handle = None
    if decisions_path:
        decisions_path.parent.mkdir(parents=True, exist_ok=True)
        if decisions_path.is_file():
            with decisions_path.open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    key = (
                        str(item.get("character_id", "")),
                        str(item.get("platform", "")),
                        str(item.get("source_comment_id", "")),
                    )
                    # A provider/network failure is not an adjudication. Keep the audit record,
                    # but let a later invocation classify the row when the provider is available.
                    if item.get("classification_status") != "classification_error":
                        existing[key] = item
        decision_handle = decisions_path.open("a", encoding="utf-8")
    pending = [
        row
        for row in rows
        if row.target_relevant is None
        and (character.character_id, row.platform, row.source_comment_id) not in existing
    ]
    if pending and llm is None:
        raise ValueError(
            "capture row lacks target_relevant; use GPT classification or adjudicate it locally"
        )
    batches = [pending[start : start + 10] for start in range(0, len(pending), 10)]

    def classify_with_isolation(batch: list[CaptureRow]) -> dict[str, dict[str, Any]]:
        classified: dict[str, dict[str, Any]] = {}
        try:
            classified = classify_relevance_batch(
                llm,
                character_name=character.character_name,
                aliases=character.aliases,
                work=character.work,
                rows=batch,
            )
        except Exception:  # noqa: BLE001 - isolate a provider-rejected item without a substitute label
            for row in batch:
                try:
                    classified[row.source_comment_id] = classify_relevance(
                        llm,
                        character_name=character.character_name,
                        aliases=character.aliases,
                        work=character.work,
                        row=row,
                    )
                except Exception as exc:  # noqa: BLE001 - fail closed and audit provider rejection
                    classified[row.source_comment_id] = {
                        "relevant": False,
                        "evidence_scope": "unclassified",
                        "rationale": "Provider rejected or failed this classification; excluded without relabeling.",
                        "classification_status": "classification_error",
                        "error_type": type(exc).__name__,
                    }
        return classified

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        results = executor.map(classify_with_isolation, batches)
        for batch, classified in zip(batches, results, strict=True):
            for row in batch:
                decision = classified[row.source_comment_id]
                decision.setdefault("classification_status", "completed")
                item = {
                    "character_id": character.character_id,
                    "source_comment_id": row.source_comment_id,
                    "platform": row.platform,
                    **decision,
                }
                key = (character.character_id, row.platform, row.source_comment_id)
                existing[key] = item
                if decision_handle:
                    decision_handle.write(json.dumps(item, ensure_ascii=False) + "\n")
                    decision_handle.flush()
    for row in rows:
        key = (character.character_id, row.platform, row.source_comment_id)
        if key in existing:
            decision = existing[key]
            relevant = bool(decision.get("relevant"))
        elif row.target_relevant is None:
            raise AssertionError("pending relevance row was not batch classified")
        else:
            relevant = row.target_relevant
            decision = {
                "relevant": relevant,
                "evidence_scope": "locally_adjudicated",
                "rationale": "Relevance supplied by the local capture audit.",
            }
        item = {
            "character_id": character.character_id,
            "source_comment_id": row.source_comment_id,
            "platform": row.platform,
            **decision,
        }
        decisions.append(item)
        if decision_handle and key not in existing:
            decision_handle.write(json.dumps(item, ensure_ascii=False) + "\n")
            decision_handle.flush()
        if not relevant:
            continue
        clean_text = redact_public_usernames(row.raw_text)
        if len(clean_text) < 12:
            continue
        comments.append(
            Comment(
                comment_id=stable_comment_id(row, character.character_id),
                character_id=character.character_id,
                character_name=character.character_name,
                work=character.work,
                platform=row.platform,
                thread_id=row.thread_id,
                author_hash=hash_author(row.platform, row.author_source_id, salt),
                timestamp=row.timestamp,
                raw_text=clean_text,
                language=row.language or detect_language(clean_text),
                source_url=row.source_url,
                collection_method=f"{row.collection_method}+local_target_relevance_audit",
                license_note=row.license_note,
                is_synthetic=False,
            )
        )
    if decision_handle:
        decision_handle.close()
    return comments, decisions


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="temporary browser-capture JSONL")
    parser.add_argument("--character-id", required=True)
    parser.add_argument("--catalog", default="data/catalog/characters.json")
    parser.add_argument("--db", default="data/corpus/comments.sqlite")
    parser.add_argument("--salt-file", default="data/private/author_salt")
    parser.add_argument("--provider", default="GPT")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--trace", default="data/private/comment_relevance_trace.jsonl")
    parser.add_argument("--decisions", default="data/audits/comment_relevance.jsonl")
    parser.add_argument("--import-log", default="data/audits/comment_imports.jsonl")
    parser.add_argument(
        "--adjudicated-only",
        action="store_true",
        help="require target_relevant on every row and make no GPT calls",
    )
    args = parser.parse_args()

    catalog = {item.character_id: item for item in load_character_catalog(args.catalog)}
    if args.character_id not in catalog:
        raise ValueError(f"unknown character_id: {args.character_id}")
    capture_path = Path(args.input)
    rows = read_capture(capture_path)
    salt = ensure_private_salt(Path(args.salt_file))
    llm = None
    if not args.adjudicated_only and any(row.target_relevant is None for row in rows):
        llm = LLMClient(
            ProviderSettings.from_env(args.provider, args.env_file), trace_path=args.trace
        )
    comments, decisions = prepare_rows(
        rows,
        character=catalog[args.character_id],
        salt=salt,
        llm=llm,
        decisions_path=Path(args.decisions),
    )
    with CommentCorpus(args.db) as corpus:
        corpus.add_characters(catalog.values())
        inserted, duplicates = corpus.add_comments(comments)
    import_record = {
        "imported_at": datetime.now(timezone.utc).isoformat(),
        "character_id": args.character_id,
        "capture_file": capture_path.name,
        "capture_sha256": hashlib.sha256(capture_path.read_bytes()).hexdigest(),
        "platforms": sorted({row.platform for row in rows}),
        "thread_ids": sorted({row.thread_id for row in rows}),
        "license_notes": sorted({row.license_note for row in rows}),
        "captured": len(rows),
        "accepted": len(comments),
        "rejected": len(rows) - len(comments),
        "inserted": inserted,
        "duplicates": duplicates,
        "classification_errors": sum(
            item.get("classification_status") == "classification_error"
            for item in decisions
        ),
        "classifier_model": llm.settings.model if llm else None,
        "decisions": str(Path(args.decisions)),
    }
    capture_metadata_path = capture_path.with_name(capture_path.name + ".meta.json")
    if capture_metadata_path.is_file():
        import_record["capture_metadata"] = json.loads(
            capture_metadata_path.read_text(encoding="utf-8")
        )
    import_log = Path(args.import_log)
    import_log.parent.mkdir(parents=True, exist_ok=True)
    with import_log.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(import_record, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "status": "ok",
                "captured": len(rows),
                "accepted": len(comments),
                "rejected": len(rows) - len(comments),
                "inserted": inserted,
                "duplicates": duplicates,
                "model": llm.settings.model if llm else None,
                "classification_errors": import_record["classification_errors"],
                "decisions": str(Path(args.decisions).resolve()),
                "import_log": str(import_log.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
