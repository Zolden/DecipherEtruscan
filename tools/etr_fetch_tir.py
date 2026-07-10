# -*- coding: utf-8 -*-
"""Fetch the open Thesaurus Inscriptionum Raeticarum (TIR) tables.

TIR is the complete scholarly digital edition of the Raetic corpus, a
Tyrsenian language and therefore the most useful external control for
Etruscan morphology.  The source MediaWiki is CC BY-SA 3.0 / GFDL.  This
script downloads only page wikitext through the public API, extracts the
top-level template parameters, and writes compact CSV tables; it never
downloads images.

Run from the repository root::

    PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe tools/etr_fetch_tir.py

The fetch is intentionally deterministic apart from upstream revisions and
the retrieval timestamp recorded in metadata.json.  Each row records the
exact MediaWiki revision id and timestamp used.
"""
from __future__ import annotations

import csv
import hashlib
import html
import json
import os
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterable
from datetime import datetime, timezone
from html.parser import HTMLParser


API = "https://tir.univie.ac.at/api.php"
SITE = "https://tir.univie.ac.at/wiki/"
OUT_DIR = os.path.join("data", "external", "tir")
USER_AGENT = (
    "DecipherEtruscan-research-fetch/1.0 "
    "(+https://github.com/Zolden/DecipherEtruscan)"
)
RETRIES = 4
BATCH = 40


def fetch_json(params: dict[str, str | int]) -> dict:
    query = urllib.parse.urlencode(params)
    req = urllib.request.Request(f"{API}?{query}", headers={"User-Agent": USER_AGENT})
    for attempt in range(RETRIES):
        try:
            with urllib.request.urlopen(req, timeout=45) as response:
                return json.load(response)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            if attempt + 1 == RETRIES:
                raise
            time.sleep(1.0 * (2**attempt))
    raise AssertionError("unreachable")


def category_titles(category: str) -> list[str]:
    titles: list[str] = []
    cont: str | None = None
    while True:
        params: dict[str, str | int] = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": f"Category:{category}",
            "cmnamespace": 0,
            "cmlimit": "max",
            "format": "json",
            "formatversion": 2,
        }
        if cont:
            params["cmcontinue"] = cont
        payload = fetch_json(params)
        titles.extend(x["title"] for x in payload["query"]["categorymembers"])
        cont = payload.get("continue", {}).get("cmcontinue")
        if not cont:
            break
    return sorted(set(titles))


def chunks(items: list[str], size: int) -> Iterable[list[str]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def page_revisions(titles: list[str]) -> dict[str, dict[str, str | int]]:
    pages: dict[str, dict[str, str | int]] = {}
    for batch in chunks(titles, BATCH):
        payload = fetch_json(
            {
                "action": "query",
                "prop": "revisions",
                "rvprop": "ids|timestamp|content",
                "rvslots": "main",
                "titles": "|".join(batch),
                "format": "json",
                "formatversion": 2,
            }
        )
        for page in payload["query"]["pages"]:
            if page.get("missing"):
                continue
            revision = page["revisions"][0]
            pages[page["title"]] = {
                "revision_id": revision["revid"],
                "revision_timestamp": revision["timestamp"],
                "wikitext": revision["slots"]["main"]["content"],
            }
        time.sleep(0.10)
    return pages


def template_params(wikitext: str, expected: str) -> dict[str, str]:
    """Parse the first top-level TIR template.

    TIR stores one parameter per physical line.  Nested templates occur inside
    values but do not start at column zero, so line parsing is safer here than
    a generic regular expression over balanced braces.
    """
    lines = wikitext.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    start = next(
        (i for i, line in enumerate(lines) if line.strip().lower() == "{{" + expected),
        None,
    )
    if start is None:
        return {}
    out: dict[str, str] = {}
    current: str | None = None
    for line in lines[start + 1 :]:
        if line.startswith("}}"):
            break
        if line.startswith("|") and "=" in line:
            key, value = line[1:].split("=", 1)
            current = key.strip()
            out[current] = value.strip()
        elif current and line.strip():
            out[current] += "\n" + line.strip()
    return out


def wiki_url(title: str) -> str:
    return SITE + urllib.parse.quote(title.replace(" ", "_"), safe="()'*-._~")


def clean_reading(value: str) -> str:
    """Remove TIR's link/control markup and retain the displayed reading.

    Most links are ``link-target!display-form``.  A link can start after a
    separator without intervening whitespace (``:unknown!p[``), and a few
    damaged readings append a second, non-displayed alternative as
    ``display!alternative``.  Other bare exclamation marks are rendering
    controls before lacunae or spaces; none is literal inscription text.
    """
    # Keep the boundary while dropping link targets.  The explicit boundary
    # avoids treating a bare rendering-control ``!`` as a link separator.
    value = re.sub(r"(^|[\s:·/])([^\s:·/!]+)!", r"\1", value)
    # TIR renders the first of two inline alternatives (e.g. AK-1.5 and
    # SL-2.2).  Drop the second before removing the remaining bare controls.
    value = re.sub(r"(?<=\S)!\S+", "", value)
    value = value.replace("!", "")
    # Ordinary whitespace separates template elements and is not necessarily
    # present in the inscription (e.g. SZ-1.1 ``φirimaθinaχe``).  Conversely,
    # ``space`` is an explicit TIR archigrapheme and raw NBSPs are displayed.
    # Protect the latter two before removing template whitespace.
    space_marker = "\ue000"
    value = re.sub(r"(?<!\S)space(?!\S)", space_marker, value)
    value = value.replace("\N{NO-BREAK SPACE}", space_marker)
    value = re.sub(r"\s+", "", value)
    return value.replace(space_marker, "\N{NO-BREAK SPACE}").strip()


class _BroadtableParser(HTMLParser):
    """Extract rows from TIR's rendered Semantic MediaWiki broadtable."""

    def __init__(self) -> None:
        super().__init__()
        self.in_table = False
        self.row: list[str] | None = None
        self.cell: list[str] | None = None
        self.rows: list[list[str]] = []

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        attributes = dict(attrs)
        if tag == "table" and "broadtable" in (attributes.get("class") or ""):
            self.in_table = True
        elif self.in_table and tag == "tr":
            self.row = []
        elif self.in_table and self.row is not None and tag in ("td", "th"):
            self.cell = []

    def handle_endtag(self, tag: str) -> None:
        if self.in_table and tag in ("td", "th") and self.cell is not None:
            assert self.row is not None
            self.row.append("".join(self.cell))
            self.cell = None
        elif self.in_table and tag == "tr" and self.row is not None:
            self.rows.append(self.row)
            self.row = None
        elif self.in_table and tag == "table":
            self.in_table = False

    def handle_data(self, data: str) -> None:
        if self.cell is not None:
            self.cell.append(data)


def rendered_inscription_readings() -> dict[str, str]:
    """Return the official rendered plain readings for parser validation."""
    payload = fetch_json(
        {
            "action": "parse",
            "page": "Category:Inscription",
            "prop": "text",
            "format": "json",
            "formatversion": 2,
        }
    )
    parser = _BroadtableParser()
    parser.feed(payload["parse"]["text"])
    return {
        row[0].strip(): unicodedata.normalize("NFC", row[1].strip())
        for row in parser.rows[1:]
        if len(row) >= 2
    }


def page_rows(category: str, template: str) -> list[dict[str, str | int]]:
    titles = category_titles(category)
    pages = page_revisions(titles)
    rows: list[dict[str, str | int]] = []
    for title in titles:
        page = pages.get(title)
        if not page:
            continue
        params = template_params(str(page["wikitext"]), template)
        if not params:
            continue
        params = {
            key: unicodedata.normalize("NFC", html.unescape(value))
            for key, value in params.items()
        }
        if template == "inscription":
            params["reading_raw"] = params.get("reading", "")
            params["reading"] = clean_reading(params.get("reading", ""))
        rows.append(
            {
                "title": title,
                **params,
                "revision_id": int(page["revision_id"]),
                "revision_timestamp": str(page["revision_timestamp"]),
                "url": wiki_url(title),
            }
        )
    return rows


def selected(row: dict[str, str | int], fields: list[str]) -> dict[str, str | int]:
    return {field: row.get(field, "") for field in fields}


def write_csv(path: str, rows: list[dict[str, str | int]], fields: list[str]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(selected(row, fields))


INSCRIPTION_FIELDS = [
    "title",
    "reading",
    "reading_raw",
    "language",
    "meaning",
    "object",
    "type_object",
    "material",
    "date",
    "sortdate",
    "date_derivation",
    "site",
    "field_name",
    "location",
    "position",
    "direction",
    "script",
    "alphabet",
    "line_number",
    "letter_number_min",
    "letter_number_max",
    "sign_number",
    "craftsmanship",
    "condition",
    "sigla_tm",
    "sigla_pid",
    "sigla_mancini",
    "sigla_mlr",
    "source",
    "object_source",
    "checklevel",
    "object_checklevel",
    "problem",
    "revision_id",
    "revision_timestamp",
    "object_revision_id",
    "object_revision_timestamp",
    "url",
    "object_url",
]

WORD_FIELDS = [
    "title",
    "type_word",
    "language",
    "lemma",
    "number",
    "case",
    "gender",
    "tense",
    "analysis_morphemic",
    "meaning",
    "checklevel",
    "revision_id",
    "revision_timestamp",
    "url",
]

MORPHEME_FIELDS = [
    "title",
    "type_morpheme",
    "language",
    "meaning",
    "function",
    "checklevel",
    "revision_id",
    "revision_timestamp",
    "url",
]

OBJECT_FIELDS = [
    "title",
    "type_object",
    "material",
    "date",
    "sortdate",
    "date_derivation",
    "site",
    "field_name",
    "location",
    "inventory_number",
    "dimension",
    "condition",
    "source",
    "checklevel",
    "revision_id",
    "revision_timestamp",
    "url",
]


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    print("Fetching TIR categories via public MediaWiki API ...")
    inscriptions = page_rows("Inscription", "inscription")
    objects = page_rows("Object", "object")
    words = page_rows("Word", "word")
    morphemes = page_rows("Morpheme", "morpheme")

    # The source template format is not a public interchange specification.
    # Compare every cleaned reading with TIR's own rendered plain-text table so
    # a future markup change fails loudly instead of creating false tokens.
    rendered = rendered_inscription_readings()
    missing_rendered = [
        str(row["title"]) for row in inscriptions if str(row["title"]) not in rendered
    ]
    reading_mismatches = [
        str(row["title"])
        for row in inscriptions
        if str(row["title"]) in rendered
        and row.get("reading", "") != rendered[str(row["title"])]
    ]
    if missing_rendered or reading_mismatches:
        raise RuntimeError(
            "TIR reading parser no longer matches rendered source: "
            f"missing={missing_rendered[:5]}, mismatches={reading_mismatches[:5]}"
        )

    objects_by_title = {str(row["title"]): row for row in objects}
    joined = 0
    for inscription in inscriptions:
        obj = objects_by_title.get(str(inscription.get("object", "")))
        if not obj:
            continue
        joined += 1
        for field in (
            "type_object",
            "material",
            "date",
            "sortdate",
            "date_derivation",
            "site",
            "field_name",
            "location",
        ):
            inscription[field] = obj.get(field, "")
        inscription["object_source"] = obj.get("source", "")
        inscription["object_checklevel"] = obj.get("checklevel", "")
        inscription["object_revision_id"] = obj.get("revision_id", "")
        inscription["object_revision_timestamp"] = obj.get(
            "revision_timestamp", ""
        )
        inscription["object_url"] = obj.get("url", "")

    outputs = {
        "tir_inscriptions.csv": (inscriptions, INSCRIPTION_FIELDS),
        "tir_objects.csv": (objects, OBJECT_FIELDS),
        "tir_words.csv": (words, WORD_FIELDS),
        "tir_morphemes.csv": (morphemes, MORPHEME_FIELDS),
    }
    for filename, (rows, fields) in outputs.items():
        write_csv(os.path.join(OUT_DIR, filename), rows, fields)

    files = {name: sha256(os.path.join(OUT_DIR, name)) for name in outputs}
    metadata = {
        "source": "Thesaurus Inscriptionum Raeticarum",
        "source_url": "https://tir.univie.ac.at/",
        "api_url": API,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "license": "CC BY-SA 3.0 and GFDL; see upstream Terms of Use",
        "license_url": "https://tir.univie.ac.at/wiki/Project:Terms_of_Use",
        "counts": {
            "inscriptions": len(inscriptions),
            "objects": len(objects),
            "words": len(words),
            "morphemes": len(morphemes),
            "inscriptions_joined_to_objects": joined,
        },
        "reading_validation": {
            "source": "rendered Category:Inscription plain-text column",
            "compared": len(inscriptions),
            "exact_matches": len(inscriptions),
        },
        "sha256": files,
    }
    metadata_path = os.path.join(OUT_DIR, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    files["metadata.json"] = sha256(metadata_path)
    with open(
        os.path.join(OUT_DIR, "SHA256SUMS"), "w", encoding="ascii", newline="\n"
    ) as handle:
        for filename in sorted(files):
            handle.write(f"{files[filename]}  {filename}\n")

    print(json.dumps(metadata["counts"], ensure_ascii=False, indent=2))
    print(f"Wrote {len(outputs)} tables to {OUT_DIR}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
