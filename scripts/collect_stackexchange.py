#!/usr/bin/env python3
"""Collect target-candidate text through the official Stack Exchange API.

The output is a temporary local capture. It is not imported until GPT/local relevance adjudication,
username redaction, and salted author hashing are complete.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import ssl
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Iterable

import certifi


API_ROOT = "https://api.stackexchange.com/2.3"
LICENSE_NOTE = (
    "CC BY-SA under the Stack Exchange content terms; canonical URL and private attribution "
    "metadata retained. Raw text is not redistributed by this project."
)


def plain_text(value: str) -> str:
    value = re.sub(r"<pre><code>.*?</code></pre>", " ", value, flags=re.DOTALL | re.IGNORECASE)
    value = re.sub(r"<[^>]+>", " ", value)
    return " ".join(html.unescape(value).split())


def author_id(item: dict[str, Any]) -> str:
    owner = item.get("owner") or {}
    if owner.get("user_id") is not None:
        return str(owner["user_id"])
    return "display:" + str(owner.get("display_name") or "unknown")


class StackExchangeAPI:
    def __init__(self, *, site: str, key: str | None = None) -> None:
        self.site = site
        self.key = key
        self.quota_remaining: int | None = None

    def page(self, endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
        values = {"site": self.site, **params}
        if self.key:
            values["key"] = self.key
        url = f"{API_ROOT}/{endpoint.lstrip('/')}?{urllib.parse.urlencode(values)}"
        request = urllib.request.Request(url, headers={"User-Agent": "SideProfileResearch/1.0"})
        context = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(request, context=context) as response:  # noqa: S310 - fixed official API root
            payload = json.load(response)
        if payload.get("quota_remaining") is not None:
            self.quota_remaining = int(payload["quota_remaining"])
        if payload.get("backoff"):
            time.sleep(int(payload["backoff"]))
        return payload

    def all_pages(self, endpoint: str, params: dict[str, Any]) -> Iterable[dict[str, Any]]:
        page = 1
        while True:
            payload = self.page(endpoint, {**params, "page": page, "pagesize": 100})
            yield from payload.get("items", [])
            if not payload.get("has_more"):
                break
            page += 1


def capture_row(
    *,
    kind: str,
    item: dict[str, Any],
    question_id: int,
    source_title: str,
    source_url: str,
) -> dict[str, Any] | None:
    text = plain_text(str(item.get("body_markdown") or item.get("body") or ""))
    if len(text) < 12:
        return None
    source_id = item.get(f"{kind}_id") or item.get("post_id")
    return {
        "platform": "stackexchange",
        "thread_id": str(question_id),
        "source_comment_id": f"{kind}:{source_id}",
        "author_source_id": author_id(item),
        "raw_text": text,
        "source_url": source_url,
        "source_title": source_title,
        "timestamp": str(item.get("creation_date", "")),
        "language": "en",
        "collection_method": "stackexchange_official_api",
        "license_note": LICENSE_NOTE,
        "target_relevant": None,
    }


def chunks(values: list[int], size: int = 100) -> Iterable[list[int]]:
    for index in range(0, len(values), size):
        yield values[index : index + size]


def collect(api: StackExchangeAPI, query: str) -> list[dict[str, Any]]:
    questions = list(
        api.all_pages(
            "search/advanced",
            {
                "q": query,
                "order": "desc",
                "sort": "relevance",
                "filter": "withbody",
            },
        )
    )
    rows: list[dict[str, Any]] = []
    question_meta = {
        int(item["question_id"]): {
            "title": plain_text(str(item.get("title", ""))),
            "link": str(item.get("link", "")),
        }
        for item in questions
    }
    for question in questions:
        question_id = int(question["question_id"])
        title = question_meta[question_id]["title"]
        link = question_meta[question_id]["link"]
        row = capture_row(
            kind="question",
            item=question,
            question_id=question_id,
            source_title=title,
            source_url=link,
        )
        if row:
            rows.append(row)
    question_ids = sorted(question_meta)
    answers: list[dict[str, Any]] = []
    for batch in chunks(question_ids):
        joined = ";".join(str(value) for value in batch)
        answers.extend(
            api.all_pages(
                f"questions/{joined}/answers",
                {"order": "asc", "sort": "creation", "filter": "withbody"},
            )
        )
        for comment in api.all_pages(
            f"questions/{joined}/comments",
            {"order": "asc", "sort": "creation", "filter": "withbody"},
        ):
            question_id = int(comment["post_id"])
            meta = question_meta[question_id]
            comment_id = int(comment["comment_id"])
            row = capture_row(
                kind="comment",
                item=comment,
                question_id=question_id,
                source_title=meta["title"],
                source_url=f"{meta['link']}#comment{comment_id}_{question_id}",
            )
            if row:
                rows.append(row)
    answer_to_question = {
        int(answer["answer_id"]): int(answer["question_id"]) for answer in answers
    }
    for answer in answers:
        question_id = int(answer["question_id"])
        meta = question_meta[question_id]
        answer_id = int(answer["answer_id"])
        row = capture_row(
            kind="answer",
            item=answer,
            question_id=question_id,
            source_title=meta["title"],
            source_url=f"{meta['link']}#answer-{answer_id}",
        )
        if row:
            rows.append(row)
    for batch in chunks(sorted(answer_to_question)):
        joined = ";".join(str(value) for value in batch)
        for comment in api.all_pages(
            f"answers/{joined}/comments",
            {"order": "asc", "sort": "creation", "filter": "withbody"},
        ):
            answer_id = int(comment["post_id"])
            question_id = answer_to_question[answer_id]
            meta = question_meta[question_id]
            comment_id = int(comment["comment_id"])
            row = capture_row(
                kind="comment",
                item=comment,
                question_id=question_id,
                source_title=meta["title"],
                source_url=f"{meta['link']}#comment{comment_id}_{answer_id}",
            )
            if row:
                rows.append(row)
    deduped = {}
    for row in rows:
        key = (row["thread_id"], row["source_comment_id"])
        deduped[key] = row
    return list(deduped.values())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--site", default="scifi")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    api = StackExchangeAPI(site=args.site, key=os.environ.get("STACKEXCHANGE_KEY"))
    rows = collect(api, args.query)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "status": "ok",
                "site": args.site,
                "query": args.query,
                "captured": len(rows),
                "quota_remaining": api.quota_remaining,
                "output": str(output.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
