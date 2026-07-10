#!/usr/bin/env python3
"""Build a compact manifest from the pinned USEP EpiDoc XML subset."""

from __future__ import annotations

import csv
import hashlib
import re
import xml.etree.ElementTree as ET
from pathlib import Path


HERE = Path(__file__).resolve().parent
NS = {"tei": "http://www.tei-c.org/ns/1.0"}


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def normalize(text: str) -> str:
    text = re.sub(r"\s*\|\|\s*", " || ", text)
    text = re.sub(r"\s*\|\s*", " | ", text)
    return " ".join(text.split())


def render(element: ET.Element) -> str:
    out = [element.text or ""]
    for child in element:
        name = local_name(child.tag)
        if name == "lb":
            out.append(" | ")
        elif name == "gap":
            extent = child.get("extent", "?")
            unit = child.get("unit", "")
            out.append(f" [gap:{extent}{(' ' + unit) if unit else ''}] ")
        elif name == "space":
            out.append(" [space] ")
        elif name == "g" and not "".join(child.itertext()).strip():
            marker = " : " if "interpunct" in " ".join(child.attrib.values()) else " [glyph] "
            out.append(marker)
        else:
            out.append(render(child))
        out.append(child.tail or "")
    return normalize("".join(out)).removeprefix("| ")


def node_text(root: ET.Element, path: str) -> str:
    node = root.find(path, NS)
    return normalize("".join(node.itertext())) if node is not None else ""


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
    fieldnames = [
        "status",
        "usep_id",
        "language",
        "other_languages",
        "tm_id",
        "date_label",
        "not_before",
        "not_after",
        "date_evidence",
        "origin_label",
        "origin_ref",
        "summary",
        "object_class",
        "material_class",
        "condition_class",
        "edition_text",
        "translation",
        "source_file",
    ]
    rows = []
    for status in ("transcribed", "metadata_only"):
        for path in sorted((HERE / status).glob("*.xml")):
            root = ET.parse(path).getroot()
            language = root.find(".//tei:textLang", NS)
            date = root.find(".//tei:origin/tei:date", NS)
            place = root.find(".//tei:origin/tei:placeName", NS)
            obj = root.find(".//tei:objectDesc", NS)
            support = root.find(".//tei:supportDesc", NS)
            condition = root.find(".//tei:condition", NS)
            editions = root.findall(".//tei:div[@type='edition']", NS)
            translations = root.findall(".//tei:div[@type='translation']", NS)
            rows.append(
                {
                    "status": status,
                    "usep_id": node_text(root, ".//tei:titleStmt/tei:title"),
                    "language": language.get("mainLang", "") if language is not None else "",
                    "other_languages": language.get("otherLangs", "") if language is not None else "",
                    "tm_id": node_text(root, ".//tei:altIdentifier[@type='TM_number']/tei:idno"),
                    "date_label": normalize("".join(date.itertext())) if date is not None else "",
                    "not_before": date.get("notBefore", "") if date is not None else "",
                    "not_after": date.get("notAfter", "") if date is not None else "",
                    "date_evidence": date.get("evidence", "") if date is not None else "",
                    "origin_label": normalize("".join(place.itertext())) if place is not None else "",
                    "origin_ref": place.get("ref", "") if place is not None else "",
                    "summary": node_text(root, ".//tei:msItem/tei:p"),
                    "object_class": obj.get("ana", "") if obj is not None else "",
                    "material_class": support.get("ana", "") if support is not None else "",
                    "condition_class": condition.get("ana", "") if condition is not None else "",
                    "edition_text": " || ".join(render(node) for node in editions),
                    "translation": " || ".join(render(node) for node in translations),
                    "source_file": path.relative_to(HERE).as_posix(),
                }
            )

    with (HERE / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    write_checksums()


if __name__ == "__main__":
    main()
