#!/usr/bin/env python3
"""Build a flat CSV from the pinned Lexicon Leponticum API responses."""

from __future__ import annotations

import csv
import hashlib
import html
import json
import re
from pathlib import Path


HERE = Path(__file__).resolve().parent


def clean(value: object) -> str:
    text = html.unescape(str(value))
    text = re.sub(r"<[^>]+>", "", text)
    return " ".join(text.replace("\xa0", " ").split())


def first(values: list[object] | None) -> object:
    return values[0] if values else ""


def write_checksums() -> None:
    lines = []
    for path in sorted(p for p in HERE.rglob("*") if p.is_file()):
        if path.name == "SHA256SUMS":
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.relative_to(HERE).as_posix()}\r\n")
    (HERE / "SHA256SUMS").write_text(
        "".join(lines), encoding="ascii", newline=""
    )


def main() -> None:
    ask = json.loads((HERE / "ask_records.json").read_text(encoding="utf-8"))
    pages = json.loads((HERE / "pages_wikitext.json").read_text(encoding="utf-8"))
    revisions = {}
    for page in pages["query"]["pages"]:
        rev = first(page.get("revisions"))
        revisions[page["title"]] = rev if isinstance(rev, dict) else {}

    fieldnames = [
        "siglum",
        "text_plain",
        "date_label",
        "sortdate",
        "latitude",
        "longitude",
        "object",
        "direction",
        "type_inscription",
        "condition",
        "meaning",
        "sources",
        "checklevel",
        "page_url",
        "revision_id",
        "revision_timestamp",
    ]
    with (HERE / "records.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for siglum, result in ask["query"]["results"].items():
            out = result["printouts"]
            coordinate = first(out.get("coordinate"))
            obj = first(out.get("object"))
            sources = out.get("source", [])
            rev = revisions.get(siglum, {})
            writer.writerow(
                {
                    "siglum": siglum,
                    "text_plain": clean(first(out.get("text plain"))),
                    "date_label": clean(first(out.get("date"))),
                    "sortdate": first(out.get("sortdate")),
                    "latitude": coordinate.get("lat", "") if isinstance(coordinate, dict) else "",
                    "longitude": coordinate.get("lon", "") if isinstance(coordinate, dict) else "",
                    "object": obj.get("fulltext", "") if isinstance(obj, dict) else "",
                    "direction": clean(first(out.get("direction"))),
                    "type_inscription": clean(first(out.get("type inscription"))),
                    "condition": "; ".join(map(clean, out.get("condition", []))),
                    "meaning": clean(first(out.get("meaning"))),
                    "sources": "; ".join(
                        source.get("fulltext", "")
                        for source in sources
                        if isinstance(source, dict)
                    ),
                    "checklevel": first(out.get("checklevel")),
                    "page_url": result["fullurl"],
                    "revision_id": rev.get("revid", ""),
                    "revision_timestamp": rev.get("timestamp", ""),
                }
            )
    write_checksums()


if __name__ == "__main__":
    main()
