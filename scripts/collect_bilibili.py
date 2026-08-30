#!/usr/bin/env python3
"""Collect public Bilibili comments into the temporary audited-capture schema.

This utility runs only on the connected preparation side. The formal experiment consumes the frozen
SQLite corpus and never calls Bilibili.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import ssl
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import certifi


API_ROOT = "https://api.bilibili.com"
MIXIN_KEY_ENC_TAB = [
    46, 47, 18, 2, 53, 8, 23, 32, 15, 50, 10, 31, 58, 3, 45, 35,
    27, 43, 5, 49, 33, 9, 42, 19, 29, 28, 14, 39, 12, 38, 41, 13,
    37, 48, 7, 16, 24, 55, 40, 61, 26, 17, 0, 1, 60, 51, 30, 4,
    22, 25, 54, 21, 56, 59, 6, 63, 57, 62, 11, 36, 20, 34, 44, 52,
]
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0 Safari/537.36",
    "Referer": "https://www.bilibili.com/",
}
LICENSE_NOTE = (
    "Publicly visible Bilibili comment retained privately for research audit; public video and "
    "comment identifiers preserved; redistribution is not authorized."
)


def request_json_url(
    url: str, *, allowed_codes: tuple[int, ...] = (0,)
) -> dict[str, Any]:
    request = urllib.request.Request(url, headers=HEADERS)
    context = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(request, context=context) as response:  # noqa: S310
        payload = json.load(response)
    if payload.get("code") not in allowed_codes:
        raise RuntimeError(f"Bilibili API failed: {payload.get('code')} {payload.get('message')}")
    return payload


def request_json(endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
    url = f"{API_ROOT}/{endpoint.lstrip('/')}?{urllib.parse.urlencode(params)}"
    return request_json_url(url)


def mixin_key() -> str:
    payload = request_json_url(
        f"{API_ROOT}/x/web-interface/nav", allowed_codes=(0, -101)
    )
    wbi = payload["data"]["wbi_img"]
    img_key = wbi["img_url"].rsplit("/", 1)[-1].split(".", 1)[0]
    sub_key = wbi["sub_url"].rsplit("/", 1)[-1].split(".", 1)[0]
    source = img_key + sub_key
    return "".join(source[index] for index in MIXIN_KEY_ENC_TAB)[:32]


def signed_url(endpoint: str, params: dict[str, Any], key: str) -> str:
    signed = {**params, "wts": int(time.time())}
    clean = {
        item_key: str(value).translate({ord(char): None for char in "!'()*"})
        for item_key, value in signed.items()
    }
    query = urllib.parse.urlencode(sorted(clean.items()))
    signature = hashlib.md5((query + key).encode()).hexdigest()  # noqa: S324
    return f"{API_ROOT}/{endpoint.lstrip('/')}?{query}&w_rid={signature}"


def reply_items(reply: dict[str, Any]) -> Iterable[dict[str, Any]]:
    yield reply
    for child in reply.get("replies") or []:
        yield from reply_items(child)


def capture_row(*, item: dict[str, Any], bvid: str, aid: int, title: str) -> dict[str, Any] | None:
    text = " ".join(str((item.get("content") or {}).get("message") or "").split())
    if not text:
        return None
    rpid = str(item.get("rpid") or item.get("rpid_str") or "")
    if not rpid:
        return None
    member = item.get("member") or {}
    author_id = str(member.get("mid") or member.get("uname") or f"anonymous:{rpid}")
    ctime = item.get("ctime")
    timestamp = (
        datetime.fromtimestamp(int(ctime), tz=timezone.utc).isoformat()
        if ctime is not None
        else ""
    )
    return {
        "platform": "bilibili",
        "thread_id": bvid,
        "source_comment_id": rpid,
        "author_source_id": author_id,
        "raw_text": text,
        "source_url": f"https://www.bilibili.com/video/{bvid}/#reply{rpid}",
        "source_title": title,
        "timestamp": timestamp,
        "language": "zh",
        "collection_method": "bilibili_public_comment_api",
        "license_note": LICENSE_NOTE,
        "target_relevant": None,
        "_aid": aid,
    }


def collect_nested(*, aid: int, root: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    page = 1
    while True:
        payload = request_json(
            "/x/v2/reply/reply",
            {"type": 1, "oid": aid, "root": root, "pn": page, "ps": 20},
        )
        data = payload.get("data") or {}
        replies = data.get("replies") or []
        rows.extend(replies)
        page_info = data.get("page") or {}
        count = int(page_info.get("count") or 0)
        size = int(page_info.get("size") or len(replies) or 20)
        if not replies or page * size >= count:
            break
        page += 1
    return rows


def collect(bvid: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    view = request_json("/x/web-interface/view", {"bvid": bvid})["data"]
    aid = int(view["aid"])
    title = str(view["title"])
    captured: dict[str, dict[str, Any]] = {}
    nested_fetch_failures = 0
    nested_endpoint_available = True
    offset = ""
    key = mixin_key()
    while True:
        pagination = json.dumps({"offset": offset}, ensure_ascii=False, separators=(",", ":"))
        payload = request_json_url(
            signed_url(
                "/x/v2/reply/wbi/main",
                {
                    "type": 1,
                    "oid": aid,
                    "mode": 3,
                    "plat": 1,
                    "pagination_str": pagination,
                    "web_location": 1315875,
                },
                key,
            )
        )
        data = payload.get("data") or {}
        for root in data.get("replies") or []:
            visible_children = list(root.get("replies") or [])
            for item in reply_items(root):
                captured[str(item.get("rpid") or item.get("rpid_str"))] = item
            if int(root.get("rcount") or 0) > len(visible_children):
                if not nested_endpoint_available:
                    nested_fetch_failures += 1
                else:
                    try:
                        nested = collect_nested(aid=aid, root=int(root["rpid"]))
                    except RuntimeError:
                        # The signed WBI main endpoint already supplies the root and its visible child
                        # replies. Some public sessions reject the legacy nested-reply endpoint; keep
                        # the visible data and report that the capture is non-exhaustive.
                        nested_endpoint_available = False
                        nested_fetch_failures += 1
                    else:
                        for child in nested:
                            captured[str(child.get("rpid") or child.get("rpid_str"))] = child
        cursor_info = data.get("cursor") or {}
        if cursor_info.get("is_end"):
            break
        next_offset = str(
            (cursor_info.get("pagination_reply") or {}).get("next_offset") or ""
        )
        if not next_offset or next_offset == offset:
            raise RuntimeError("Bilibili WBI comment cursor did not advance")
        offset = next_offset
    rows = []
    for item in captured.values():
        row = capture_row(item=item, bvid=bvid, aid=aid, title=title)
        if row:
            row.pop("_aid", None)
            rows.append(row)
    view["_nested_fetch_failures"] = nested_fetch_failures
    return view, rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bvid", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    view, rows = collect(args.bvid)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "status": "ok",
        "bvid": args.bvid,
        "aid": view["aid"],
        "title": view["title"],
        "reported_replies": view.get("stat", {}).get("reply"),
        "captured": len(rows),
        "nested_fetch_failures": view.get("_nested_fetch_failures", 0),
        "capture_complete": view.get("_nested_fetch_failures", 0) == 0,
        "capture_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "output": str(output.resolve()),
    }
    metadata = output.with_name(output.name + ".meta.json")
    metadata.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    summary["metadata"] = str(metadata.resolve())
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
