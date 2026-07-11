# -*- coding: utf-8 -*-
"""Independent statistical audit for the 2026-07-10 SOL review.

This script deliberately does not overwrite any existing project result.  It
quantifies corpus units/uncertainty, re-tests concept convergence with a
coverage-preserving family-wise null, evaluates semantic labels across
provenance sources, audits suffix-family multiplicity, and measures exact-type
leakage in the Lemnian language classifier.

Run from the repository root:
  PYTHONIOENCODING=utf-8 PYTHONHASHSEED=0 .venv/Scripts/python.exe \
      tools/etr_method_audit_sol.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import pickle
import re
import sys
from collections import Counter, defaultdict
from itertools import combinations

import numpy as np
from scipy.stats import hypergeom

sys.stdout.reconfigure(encoding="utf-8")

SEED = 20260710
R_CONCEPT = 2000
R_SEMANTIC = 500
R_CLUSTER = 2000
R_CONTEXT = 2000
OUT_JSON = os.path.join("results", "method_audit_sol_20260710.json")
OUT_LOG = os.path.join("logs", "etr_method_audit_sol_20260710.log")
LOG: list[str] = []


def log(message: str = "") -> None:
    print(message)
    LOG.append(message)


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_view(corpus: dict) -> list[dict]:
    return [
        r
        for r in corpus["records"]
        if r["lang"] == "etr"
        and r["kind"] == "text"
        and "forgery?" not in r["flags"]
        and r.get("variant_of") is None
    ]


def words(rec: dict) -> list[str]:
    return [t["ascii"] for t in rec["toks"] if t["kind"] == "W"]


def source_words(rec: dict) -> list[str]:
    """Normalized source transcription; required for morphology."""
    return [t["form"] for t in rec["toks"] if t["kind"] == "W"]


def artifact_id(rec: dict) -> str:
    """Best available physical-object cluster; explicitly only a proxy."""
    # Known cross-edition aliases must be resolved before generic TM/CIE
    # identifiers, otherwise one physical text is counted as independent
    # support in ETP, CIEP, and CIEW.
    if (
        (rec["src"] == "CIEW" and rec["eid"] == "9001")
        or (rec["src"] == "CIEP" and rec["eid"] == "15910")
        or (rec["src"] == "ETP" and rec["eid"].startswith("LL"))
    ):
        return "MONUMENT:Liber_Linteus"
    if (
        (rec["src"] == "CIEW" and rec["eid"] == "7002")
        or (rec["src"] == "CIEP" and rec["eid"] == "8682")
    ):
        return "MONUMENT:Tabula_Capuana"
    xref = rec.get("xref") or {}
    tm = (xref.get("tm") or "").strip()
    if tm:
        return "TM:" + tm
    if rec["src"] in ("CIEP", "CIEW-CIE"):
        return "CIE:" + rec["eid"]
    if rec["src"] == "CIEW":
        return "CIEW:" + rec["eid"]
    return f'{rec["src"]}:{rec["eid"]}'


def bh_adjust(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p)
    ranked = p[order] * m / np.arange(1, m + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m, dtype=float)
    out[order] = np.minimum(ranked, 1.0)
    return out


def corpus_audit(corpus: dict, view: list[dict]) -> dict:
    out: dict = {}
    ws_all = [w for r in view for w in words(r)]
    surface_words = [
        t["form"] for r in view for t in r["toks"] if t["kind"] == "W"
    ]
    surface_counts = Counter(surface_words)
    ascii_counts = Counter(ws_all)
    clusters = defaultdict(list)
    for r in view:
        clusters[artifact_id(r)].append(r)

    by_source = {}
    for src in sorted({r["src"] for r in view}):
        rr = [r for r in view if r["src"] == src]
        lens = [len(words(r)) for r in rr]
        translated = [r for r in rr if r["trs"]]
        by_source[src] = {
            "records": len(rr),
            "artifact_proxy_clusters": len({artifact_id(r) for r in rr}),
            "tokens": int(sum(lens)),
            "single_word_records": int(sum(n == 1 for n in lens)),
            "translated_records": len(translated),
            "translated_single_word": int(sum(len(words(r)) == 1 for r in translated)),
        }

    seq_groups: dict[tuple[str, ...], list[dict]] = defaultdict(list)
    for r in view:
        seq = tuple(words(r))
        if len(seq) >= 2:
            seq_groups[seq].append(r)
    dup_groups = [g for g in seq_groups.values() if len(g) >= 2]
    cross_source = [g for g in dup_groups if len({r["src"] for r in g}) >= 2]
    cross_artifact = [g for g in dup_groups if len({artifact_id(r) for r in g}) >= 2]

    record_flags = Counter(f for r in view for f in r["flags"])
    token_flags = Counter(f for r in view for t in r["toks"] for f in t["flags"])
    uncertain_token_flags = {"damaged", "uncertain", "restored", "emended"}
    status: dict[str, list[bool]] = defaultdict(list)
    uncertain_tokens = 0
    editorial_tokens = 0
    for r in view:
        rec_uncertain = any(f.startswith("ocr") for f in r["flags"])
        for t in r["toks"]:
            if t["kind"] != "W":
                continue
            fs = set(t["flags"])
            unc = rec_uncertain or bool(fs & uncertain_token_flags)
            edit = unc or bool(fs & {"expanded", "scribal_extra", "braced"})
            uncertain_tokens += int(unc)
            editorial_tokens += int(edit)
            status[t["ascii"]].append(unc)

    ascii_forms: dict[str, set[str]] = defaultdict(set)
    for r in view:
        for t in r["toks"]:
            if t["kind"] == "W":
                ascii_forms[t["ascii"]].add(t["form"])
    collisions = {a: fs for a, fs in ascii_forms.items() if len(fs) > 1}
    only_uncertain = {w for w, observations in status.items() if all(observations)}
    downstream_uncertain = {}
    for path in (
        os.path.join("results", "semantic_hypotheses_v2.csv"),
        os.path.join("results", "concept_fog_v1.csv"),
    ):
        if os.path.exists(path):
            rows = list(csv.DictReader(open(path, encoding="utf-8")))
            hit = sum((row.get("word") or "") in only_uncertain for row in rows)
            downstream_uncertain[path.replace("\\", "/")] = {
                "rows": len(rows), "only_uncertain_rows": hit,
                "share": hit / max(len(rows), 1),
            }

    ciep = [r for r in view if r["src"] == "CIEP"]
    translated = [r for r in view if r["trs"]]
    known_alias_groups = {}
    alias_rules = {
        "Liber_Linteus": lambda r: (
            (r["src"] == "CIEW" and r["eid"] == "9001")
            or (r["src"] == "CIEP" and r["eid"] == "15910")
            or (r["src"] == "ETP" and r["eid"].startswith("LL"))
        ),
        "Tabula_Capuana": lambda r: (
            (r["src"] == "CIEW" and r["eid"] == "7002")
            or (r["src"] == "CIEP" and r["eid"] == "8682")
        ),
    }
    for monument, predicate in alias_rules.items():
        rr = [r for r in view if predicate(r)]
        known_alias_groups[monument] = {
            "records": len(rr),
            "tokens": sum(len(words(r)) for r in rr),
            "sources": dict(Counter(r["src"] for r in rr)),
            "source_eids": sorted({f'{r["src"]}:{r["eid"]}' for r in rr}),
        }
    out.update(
        {
            "records": len(view),
            "tokens": len(ws_all),
            # Keep the legacy key explicit: matching screens use the ASCII
            # projection, morphology uses source forms, and headlines report
            # both. This prevents an apparent 7152/7384 conflict.
            "types": len(ascii_counts),
            "types_measure": "ASCII-projected forms",
            "ascii_types": len(ascii_counts),
            "ascii_hapax_types": int(sum(n == 1 for n in ascii_counts.values())),
            "surface_types": len(surface_counts),
            "surface_hapax_types": int(sum(n == 1 for n in surface_counts.values())),
            "artifact_proxy_clusters": len(clusters),
            "single_word_records": int(sum(len(words(r)) == 1 for r in view)),
            "translated_records": len(translated),
            "translated_single_word": int(sum(len(words(r)) == 1 for r in translated)),
            "ciep_records": len(ciep),
            "ciep_unique_eid": len({r["eid"] for r in ciep}),
            "ciep_unique_tm": len(
                {
                    (r.get("xref") or {}).get("tm")
                    for r in ciep
                    if (r.get("xref") or {}).get("tm")
                }
            ),
            "max_records_per_artifact_proxy": max(map(len, clusters.values())),
            "exact_multiword_duplicate_groups": len(dup_groups),
            "exact_multiword_duplicate_records": int(sum(len(g) for g in dup_groups)),
            "cross_source_exact_multiword_groups": len(cross_source),
            "cross_artifact_exact_multiword_groups": len(cross_artifact),
            "cross_source_examples": [
                {
                    "text": " ".join(words(g[0])),
                    "records": [r["rid"] for r in g[:8]],
                }
                for g in sorted(cross_source, key=len, reverse=True)[:10]
            ],
            "uncertain_tokens": uncertain_tokens,
            "uncertain_token_share": uncertain_tokens / max(len(ws_all), 1),
            "editorially_marked_tokens": editorial_tokens,
            "editorially_marked_token_share": editorial_tokens / max(len(ws_all), 1),
            "types_only_in_uncertain_tokens": int(sum(all(v) for v in status.values())),
            "downstream_rows_only_uncertain": downstream_uncertain,
            "ascii_collision_types": len(collisions),
            "ascii_collision_examples": {
                a: sorted(fs)[:12]
                for a, fs in sorted(collisions.items(), key=lambda x: (-len(x[1]), x[0]))[:15]
            },
            "record_flags": dict(record_flags.most_common()),
            "token_flags": dict(token_flags.most_common()),
            "by_source": by_source,
            "known_cross_source_monument_aliases": known_alias_groups,
        }
    )

    log("=== 1. Corpus units, duplication, and uncertainty ===")
    log(
        f'canonical view: {out["records"]} records, {out["tokens"]} tokens, '
        f'{out["surface_types"]} source-form / {out["ascii_types"]} ASCII types, '
        f'but only {out["artifact_proxy_clusters"]} '
        "best-available artifact clusters"
    )
    log(
        f'CIEP: {out["ciep_records"]} concordance rows -> '
        f'{out["ciep_unique_eid"]} CIE ids / {out["ciep_unique_tm"]} known TM ids'
    )
    log(
        f'single-word records: {out["single_word_records"]}/{out["records"]} '
        f'({out["single_word_records"]/out["records"]:.1%}); translated singletons: '
        f'{out["translated_single_word"]}/{out["translated_records"]} '
        f'({out["translated_single_word"]/out["translated_records"]:.1%})'
    )
    log(
        f'exact duplicated multiword sequences: {len(dup_groups)} groups / '
        f'{sum(len(g) for g in dup_groups)} records; cross-source {len(cross_source)}, '
        f'cross-artifact-proxy {len(cross_artifact)}'
    )
    log(
        f'epistemically uncertain tokens: {uncertain_tokens}/{len(ws_all)} '
        f'({uncertain_tokens/max(len(ws_all),1):.1%}); types attested only in '
        f'uncertain tokens: {out["types_only_in_uncertain_tokens"]}'
    )
    log(
        f'ascii projection merges distinct source forms for {len(collisions)} types '
        "(expected normalization, but uncertainty must be propagated)"
    )
    for monument, row in known_alias_groups.items():
        log(
            f'known same-monument aliases {monument}: {row["records"]} records / '
            f'{row["tokens"]} tokens across {row["sources"]}'
        )
    log()
    return out


# --- Concept convergence ---------------------------------------------------
DIGRAPH = {
    "sh": "S", "th": "T", "kh": "K", "ch": "K", "ph": "P",
    "ts": "S", "dj": "S", "tj": "S", "ng": "N",
}
SINGLE = {
    "p": "P", "b": "P", "f": "P", "v": "W", "w": "W",
    "t": "T", "d": "T", "k": "K", "g": "K", "q": "K", "c": "K",
    "x": "K", "s": "S", "z": "S", "m": "M", "n": "N", "l": "L",
    "r": "R", "j": "J", "y": "J", "h": "H",
}
LANGS = ["grc", "lat", "hit", "akk", "heb", "egy", "sum"]


def skeleton(form: str) -> str:
    f = re.sub(r"[^a-z]", "", form.lower().strip().strip("?"))
    result = []
    i = 0
    while i < len(f):
        if f[i : i + 2] in DIGRAPH:
            result.append(DIGRAPH[f[i : i + 2]])
            i += 2
        else:
            if f[i] in SINGLE:
                result.append(SINGLE[f[i]])
            i += 1
    return "".join(result)


def lev(a: str, b: str) -> int:
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(
                min(previous[j] + 1, current[-1] + 1, previous[j - 1] + (ca != cb))
            )
        previous = current
    return previous[-1]


def similarity(a: str, b: str) -> float:
    return 0.0 if not a or not b else 1.0 - lev(a, b) / max(len(a), len(b))


def conv_score(forms: dict[str, list[str]]) -> int:
    score = 0
    languages = sorted(forms)
    for a, b in combinations(languages, 2):
        best = max((similarity(x, y) for x in forms[a] for y in forms[b]), default=0.0)
        score += best >= 0.7
    return score


def concept_audit() -> dict:
    rows = list(csv.DictReader(open(os.path.join("data", "concepts", "concept_lexicon.csv"), encoding="utf-8")))
    concepts = []
    for row in rows:
        forms = {}
        for language in LANGS:
            values = []
            for form in (row.get(language) or "").split("/"):
                sk = skeleton(form)
                if len(sk) >= 2:
                    values.append(sk)
            if values:
                forms[language] = values
        concepts.append({"id": row["id"], "en": row["gloss_en"], "forms": forms})
    n = len(concepts)
    observed = np.array([conv_score(c["forms"]) for c in concepts], dtype=np.int16)

    rng = np.random.default_rng(SEED)
    null = np.zeros((R_CONCEPT, n), dtype=np.int16)
    # Keep both the language-coverage mask and the number of alternative
    # forms in every populated cell.  The production null preserves neither.
    present_by_count = {}
    for lg in LANGS:
        groups: dict[int, list[int]] = defaultdict(list)
        for i, concept in enumerate(concepts):
            if lg in concept["forms"]:
                groups[len(concept["forms"][lg])].append(i)
        present_by_count[lg] = groups
    for iteration in range(R_CONCEPT):
        shuffled: list[dict[str, list[str]]] = [dict() for _ in concepts]
        for language in LANGS:
            for idx in present_by_count[language].values():
                permutation = rng.permutation(len(idx))
                for slot, value_i in zip(idx, permutation):
                    shuffled[slot][language] = concepts[idx[int(value_i)]]["forms"][language]
        null[iteration] = [conv_score(f) for f in shuffled]

    raw = ((null >= observed[None, :]).sum(axis=0) + 1) / (R_CONCEPT + 1)
    q = bh_adjust(raw)
    max_null = null.max(axis=1)
    fwer = ((max_null[:, None] >= observed[None, :]).sum(axis=0) + 1) / (R_CONCEPT + 1)
    eligible = observed >= 2
    top_order = sorted(
        range(n), key=lambda i: (-int(observed[i]), float(raw[i]), concepts[i]["en"])
    )
    top = [
        {
            "id": concepts[i]["id"],
            "gloss_en": concepts[i]["en"],
            "score": int(observed[i]),
            "n_languages": len(concepts[i]["forms"]),
            "p_conditional": float(raw[i]),
            "q_bh_290": float(q[i]),
            "p_fwer_max_290": float(fwer[i]),
        }
        for i in top_order[:25]
    ]
    out = {
        "concepts": n,
        "permutations": R_CONCEPT,
        "observed_score_ge_2": int(eligible.sum()),
        "null_mean_score_ge_2": float((null >= 2).sum(axis=1).mean()),
        "null_q95_score_ge_2": float(np.quantile((null >= 2).sum(axis=1), 0.95)),
        "global_count_ge_observed_p": float(
            (((null >= 2).sum(axis=1) >= int(eligible.sum())).sum() + 1) / (R_CONCEPT + 1)
        ),
        "conditional_p_lt_05_and_score_ge_2": int((eligible & (raw < 0.05)).sum()),
        "bh_q_lt_05_and_score_ge_2": int((eligible & (q < 0.05)).sum()),
        "fwer_p_lt_05_and_score_ge_2": int((eligible & (fwer < 0.05)).sum()),
        "top": top,
    }
    log("=== 2. Concept convergence: corrected null and multiplicity ===")
    log(
        f'observed concepts with score>=2: {out["observed_score_ge_2"]}; '
        f'coverage-preserving null mean {out["null_mean_score_ge_2"]:.1f}, '
        f'95% quantile {out["null_q95_score_ge_2"]:.0f}, '
        f'global-count p={out["global_count_ge_observed_p"]:.4f}'
    )
    log(
        f'conditional raw p<.05: {out["conditional_p_lt_05_and_score_ge_2"]}; '
        f'BH q<.05 across 290: {out["bh_q_lt_05_and_score_ge_2"]}; '
        f'max-stat FWER<.05: {out["fwer_p_lt_05_and_score_ge_2"]}'
    )
    for row in top[:8]:
        log(
            f'  {row["gloss_en"]:<18} score={row["score"]} langs={row["n_languages"]} '
            f'p={row["p_conditional"]:.4f} q={row["q_bh_290"]:.4f} '
            f'p_FWER={row["p_fwer_max_290"]:.4f}'
        )
    log()
    return out


# --- Cross-provenance semantic evaluation ---------------------------------
def gloss_class(g: str) -> str:
    g2 = g.replace("mrs-", "###-").replace("ms-", "###-")
    i_f = min([i for i in (g.find("mrs-"), g.find("ms-")) if i != -1], default=-1)
    i_m = g2.find("mr-")
    if i_f != -1 and (i_m == -1 or i_f < i_m):
        return "NAME-F"
    if i_m != -1:
        return "NAME-M"
    if re.search(r"\bgod|goddess|deit|divine", g):
        return "THEO"
    if re.search(r"\w+ed\b|\bgave\b|\bbuilt\b|\bmade\b|\bwrote\b|\bis\b|\bwas\b", g):
        return "VERB"
    return "OTHER"


def to_ascii_word(w: str | None) -> str:
    w = re.sub(r"[^a-zθχφσςśšê']", "", (w or "").strip().lower())
    table = {"θ": "th", "χ": "ch", "φ": "ph", "σ": "s", "ς": "s", "ś": "s", "š": "s", "ê": "e", "'": ""}
    return "".join(table.get(c, c) for c in w)


def semantic_labels(view: list[dict]) -> tuple[dict[str, str], dict[str, str]]:
    votes: dict[str, Counter] = defaultdict(Counter)
    for r in view:
        ws = words(r)
        if r["src"] == "CIEP" and r["trs"] and len(ws) == 1 and len(ws[0]) >= 3 and "-" not in ws[0]:
            votes[ws[0]][gloss_class(" ".join(r["trs"]).lower())] += 1
    hill = {}
    for w, cnt in votes.items():
        top = cnt.most_common()
        if len(top) == 1 or top[0][1] > top[1][1]:
            hill[w] = top[0][0]

    etp_votes: dict[str, Counter] = defaultdict(Counter)
    with open(os.path.join("data", "ETP_POS.csv"), encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            w = to_ascii_word(row.get("Etruscan"))
            if len(w) < 3:
                continue
            tag = (row.get("TAG") or "").strip()
            label = None
            if (row.get("theo") or "").strip() == "1":
                label = "THEO"
            elif tag == "VERB":
                label = "VERB"
            elif tag in ("PRON", "DET", "ADP", "PRT"):
                label = "FUNC"
            elif (row.get("masc") or "").strip() == "1":
                label = "NAME-M"
            elif (row.get("fem") or "").strip() == "1":
                label = "NAME-F"
            if label:
                etp_votes[w][label] += 1
    etp = {}
    for w, cnt in etp_votes.items():
        top = cnt.most_common()
        # Conflicting ETP_POS rows are not resolved by an analysis-chosen
        # class priority: tied gold labels are ambiguous and must be excluded.
        if len(top) == 1 or top[0][1] > top[1][1]:
            etp[w] = top[0][0]
    return hill, etp


def form_features(w: str) -> set[str]:
    f = set()
    for k in (1, 2, 3):
        if len(w) > k:
            f.add(f"s{k}:{w[-k:]}")
    for k in (1, 2):
        if len(w) > k:
            f.add(f"p{k}:{w[:k]}")
    for i in range(len(w) - 1):
        f.add("b:" + w[i : i + 2])
    f.add("l:" + str(min(len(w), 10)))
    return f


def semantic_transfer(train_map: dict[str, str], test_map: dict[str, str], rng: np.random.Generator) -> dict:
    classes = sorted(set(train_map.values()) & set(test_map.values()))
    train_words = sorted(w for w, c in train_map.items() if c in classes and w not in test_map)
    test_words = sorted(w for w, c in test_map.items() if c in classes and w not in train_map)
    class_index = {c: i for i, c in enumerate(classes)}
    feature_index: dict[str, int] = {}
    for w in train_words:
        for feature in form_features(w):
            feature_index.setdefault(feature, len(feature_index))

    def matrix(word_list: list[str]) -> np.ndarray:
        X = np.zeros((len(word_list), len(feature_index)), dtype=np.int8)
        for i, w in enumerate(word_list):
            for feature in form_features(w):
                j = feature_index.get(feature)
                if j is not None:
                    X[i, j] = 1
        return X

    Xtr, Xte = matrix(train_words), matrix(test_words)
    ytr = np.array([class_index[train_map[w]] for w in train_words])
    yte = np.array([class_index[test_map[w]] for w in test_words])
    n_classes = len(classes)

    def predict(labels: np.ndarray) -> np.ndarray:
        prior = np.log(np.bincount(labels, minlength=n_classes) + 1)
        l1 = np.zeros((n_classes, Xtr.shape[1]))
        l0 = np.zeros_like(l1)
        for c in range(n_classes):
            Xc = Xtr[labels == c]
            n1 = Xc.sum(axis=0)
            l1[c] = np.log((n1 + 1) / (len(Xc) + 2))
            l0[c] = np.log((len(Xc) - n1 + 1) / (len(Xc) + 2))
        lp = prior[None, :] + Xte @ (l1 - l0).T + l0.sum(axis=1)[None, :]
        return lp.argmax(axis=1)

    pred = predict(ytr)
    accuracy = float((pred == yte).mean())
    recalls = []
    f1s = []
    confusion = np.zeros((n_classes, n_classes), dtype=int)
    for truth, guess in zip(yte, pred):
        confusion[truth, guess] += 1
    for c in range(n_classes):
        tp = confusion[c, c]
        fn = confusion[c].sum() - tp
        fp = confusion[:, c].sum() - tp
        recall = tp / max(tp + fn, 1)
        precision = tp / max(tp + fp, 1)
        recalls.append(recall)
        f1s.append(2 * precision * recall / max(precision + recall, 1e-12))
    null = np.zeros(R_SEMANTIC)
    for i in range(R_SEMANTIC):
        yp = ytr[rng.permutation(len(ytr))]
        null[i] = (predict(yp) == yte).mean()
    majority = int(np.bincount(ytr).argmax())
    return {
        "classes": classes,
        "n_train_exclusive": len(train_words),
        "n_test_exclusive": len(test_words),
        "accuracy": accuracy,
        "majority_accuracy": float((yte == majority).mean()),
        "balanced_accuracy": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1s)),
        "permutation_null_mean": float(null.mean()),
        "permutation_p": float(((null >= accuracy).sum() + 1) / (R_SEMANTIC + 1)),
        "confusion": confusion.tolist(),
    }


def strip_case(w: str) -> str:
    for ending in ("isa", "ial", "thi", "al", "us", "sa", "ei", "ce", "s", "l"):
        if w.endswith(ending) and len(w) - len(ending) >= 3:
            return w[: -len(ending)]
    return w


def semantic_audit(view: list[dict]) -> dict:
    hill, etp = semantic_labels(view)
    corpus_types = {w for r in view for w in words(r)}
    etp = {w: c for w, c in etp.items() if w in corpus_types}
    common_classes = {"NAME-M", "NAME-F", "THEO", "VERB"}
    overlap = sorted(w for w in set(hill) & set(etp) if hill[w] in common_classes and etp[w] in common_classes)
    agreements = sum(hill[w] == etp[w] for w in overlap)
    rng = np.random.default_rng(SEED + 1)
    hill4 = {w: c for w, c in hill.items() if c in common_classes}
    etp4 = {w: c for w, c in etp.items() if c in common_classes}
    h_to_e = semantic_transfer(hill4, etp4, rng)
    e_to_h = semantic_transfer(etp4, hill4, rng)

    pooled = dict(hill)
    pooled.update(etp)
    classes = sorted(set(pooled.values()))
    words_lab = sorted(pooled)
    y = np.array([classes.index(pooled[w]) for w in words_lab])
    test = np.zeros(len(y), dtype=bool)
    split_rng = np.random.default_rng(42)
    for c in range(len(classes)):
        idx = np.where(y == c)[0]
        idx = idx[split_rng.permutation(len(idx))]
        test[idx[: max(1, len(idx) // 5)]] = True
    train_stems = {strip_case(words_lab[i]) for i in np.where(~test)[0]}
    leaked = [strip_case(words_lab[i]) in train_stems for i in np.where(test)[0]]

    out = {
        "hill_ciep_labels": len(hill),
        "etp_pos_labels_in_corpus": len(etp),
        "four_class_exact_overlap": len(overlap),
        "four_class_overlap_agreement": agreements / max(len(overlap), 1),
        "hill_to_etp_exclusive": h_to_e,
        "etp_to_hill_exclusive": e_to_h,
        "project_random_split_test_types": len(leaked),
        "project_random_split_stem_proxy_in_train": int(sum(leaked)),
        "project_random_split_stem_proxy_leak_share": float(np.mean(leaked)),
    }
    log("=== 3. Supervised semantics: cross-provenance evaluation ===")
    log(
        f'Hill/CIEP labels: {len(hill)}; ETP_POS labels in corpus: {len(etp)}; '
        f'4-class overlap {len(overlap)}, agreement {out["four_class_overlap_agreement"]:.1%}'
    )
    log(
        f'Hill-only -> ETP-only: n={h_to_e["n_test_exclusive"]}, '
        f'acc={h_to_e["accuracy"]:.1%}, balanced={h_to_e["balanced_accuracy"]:.1%}, '
        f'macro-F1={h_to_e["macro_f1"]:.3f}, p={h_to_e["permutation_p"]:.4f}'
    )
    log(
        f'ETP-only -> Hill-only: n={e_to_h["n_test_exclusive"]}, '
        f'acc={e_to_h["accuracy"]:.1%}, balanced={e_to_h["balanced_accuracy"]:.1%}, '
        f'macro-F1={e_to_h["macro_f1"]:.3f}, p={e_to_h["permutation_p"]:.4f}'
    )
    log(
        f'in the project-style fixed random split, {sum(leaked)}/{len(leaked)} '
        f'({np.mean(leaked):.1%}) test types share a crude case-stripped stem with train'
    )
    log()
    return out


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total == 0:
        return [0.0, 1.0]
    p = successes / total
    den = 1 + z * z / total
    center = (p + z * z / (2 * total)) / den
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / den
    return [center - half, center + half]


def operator_validation_audit(view: list[dict]) -> dict:
    # Imported registry only; importing the module has no analysis side effect.
    try:
        from tools.etr_operators import OPERATORS
    except ModuleNotFoundError:  # direct execution: sys.path[0] == tools/
        from etr_operators import OPERATORS

    subsets = {
        "ETP_full_translations": [r for r in view if r["src"] == "ETP" and r["trs"]],
        "CIEP_singleton_glosses": [
            r for r in view if r["src"] == "CIEP" and r["trs"] and len(words(r)) == 1
        ],
        "CIEP_multiword": [
            r for r in view if r["src"] == "CIEP" and r["trs"] and len(words(r)) >= 2
        ],
        "all_multiword": [r for r in view if r["trs"] and len(words(r)) >= 2],
    }
    output = {}
    for subset_name, recs in subsets.items():
        translations = [" ".join(r["trs"]).lower() for r in recs]
        tests = []
        for name, forms, gloss, cls, keyword in OPERATORS:
            if not keyword:
                continue
            fs = set(forms)
            has_op = np.array([any(w in fs for w in words(r)) for r in recs])
            n_op = int(has_op.sum())
            if n_op < 3:
                continue
            pattern = re.compile(keyword)
            has_kw = np.array([bool(pattern.search(t)) for t in translations])
            both = int((has_op & has_kw).sum())
            n_kw = int(has_kw.sum())
            p = float(hypergeom.sf(both - 1, len(recs), n_kw, n_op))
            tests.append(
                {
                    "operator": name,
                    "n_records": len(recs),
                    "n_operator": n_op,
                    "n_keyword": n_kw,
                    "both": both,
                    "precision": both / n_op,
                    "precision_wilson95": wilson_interval(both, n_op),
                    "base_rate": n_kw / max(len(recs), 1),
                    "risk_ratio_precision_over_base": (both / n_op) / max(n_kw / max(len(recs), 1), 1e-12),
                    "p_hypergeom": p,
                }
            )
        output[subset_name] = {
            "records": len(recs),
            "tests": tests,
            "bonferroni_significant": int(sum(min(1.0, x["p_hypergeom"] * len(tests)) < 0.05 for x in tests)),
            "inferential_status": (
                "descriptive extraction/translation QA only: record-level units, "
                "overlapping provenance subsets, and no artifact clustering"
            ),
            "multiple_testing_scope": "Bonferroni only within this subset",
        }

    log("=== 3b. Operator/gloss association split by translation provenance ===")
    for name, block in output.items():
        log(
            f'  {name}: records={block["records"]}, tests={len(block["tests"])}, '
            f'Bonferroni-significant={block["bonferroni_significant"]}'
        )
        for row in block["tests"]:
            if row["operator"] in {"mi", "clan", "lupu", "turce", "muluvanice", "tular"}:
                log(
                    f'    {row["operator"]:<12} {row["both"]}/{row["n_operator"]} '
                    f'precision={row["precision"]:.0%} base={row["base_rate"]:.1%} '
                    f'RR={row["risk_ratio_precision_over_base"]:.1f} p={row["p_hypergeom"]:.3g}'
                )
    log()
    return output


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> tuple[float, float, float]:
    confusion = np.zeros((n_classes, n_classes), dtype=int)
    for truth, guess in zip(y_true, y_pred):
        confusion[int(truth), int(guess)] += 1
    recalls = []
    f1s = []
    for c in range(n_classes):
        tp = confusion[c, c]
        fn = confusion[c].sum() - tp
        fp = confusion[:, c].sum() - tp
        recall = tp / max(tp + fn, 1)
        precision = tp / max(tp + fp, 1)
        recalls.append(recall)
        f1s.append(2 * precision * recall / max(precision + recall, 1e-12))
    return float((y_true == y_pred).mean()), float(np.mean(recalls)), float(np.mean(f1s))


def distributional_anchor_audit(view: list[dict]) -> dict:
    """A small, source-clean context-only semantic experiment.

    CIEP concordance rows are excluded.  ETP_POS supplies independent gold;
    word form is never a feature.  Case-stripped families stay in one fold.
    """
    clean_records = [
        r for r in view
        if r["src"] != "CIEP" and len(words(r)) >= 2
    ]
    frequency = Counter(w for r in clean_records for w in words(r) if "-" not in w and len(w) >= 3)
    _, etp = semantic_labels(view)
    allowed = {"NAME-M", "NAME-F", "THEO", "VERB", "FUNC"}
    labels = {w: c for w, c in etp.items() if c in allowed and frequency[w] >= 2}
    row_words = sorted(labels)
    row_index = {w: i for i, w in enumerate(row_words)}
    context_words = [w for w, _ in frequency.most_common(500)]
    context_index = {w: i for i, w in enumerate(context_words)}
    counts = np.zeros((len(row_words), len(context_words)), dtype=float)
    for rec in clean_records:
        ws = [w for w in words(rec) if "-" not in w and len(w) >= 3]
        for i, target in enumerate(ws):
            ri = row_index.get(target)
            if ri is None:
                continue
            for j in range(max(0, i - 2), min(len(ws), i + 3)):
                if j == i:
                    continue
                ci = context_index.get(ws[j])
                if ci is not None:
                    counts[ri, ci] += 1
    total = counts.sum()
    row_sum = counts.sum(axis=1, keepdims=True)
    col_sum = counts.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        ppmi = np.maximum(np.log((counts * total) / (row_sum @ col_sum)), 0.0)
    ppmi[~np.isfinite(ppmi)] = 0.0
    norms = np.linalg.norm(ppmi, axis=1, keepdims=True)
    X = ppmi / np.maximum(norms, 1e-12)

    classes = sorted(set(labels.values()))
    ci = {c: i for i, c in enumerate(classes)}
    y = np.array([ci[labels[w]] for w in row_words])
    # Family-blocked folds.  The assignment is deterministic and independent
    # of labels, avoiding the fixed random type split used by v1/v2.
    families = sorted({strip_case(w) for w in row_words})
    rng = np.random.default_rng(SEED + 4)
    family_order = [families[i] for i in rng.permutation(len(families))]
    fold_of = {family: i % 5 for i, family in enumerate(family_order)}
    folds = np.array([fold_of[strip_case(w)] for w in row_words])

    def crossval(labels_array: np.ndarray) -> np.ndarray:
        pred = np.zeros(len(y), dtype=int)
        for fold in range(5):
            train = folds != fold
            test = ~train
            centroids = np.zeros((len(classes), X.shape[1]))
            for c in range(len(classes)):
                rows_c = X[train & (labels_array == c)]
                if len(rows_c):
                    centroids[c] = rows_c.mean(axis=0)
            centroids /= np.maximum(np.linalg.norm(centroids, axis=1, keepdims=True), 1e-12)
            pred[test] = (X[test] @ centroids.T).argmax(axis=1)
        return pred

    pred = crossval(y)
    accuracy, balanced, macro_f1 = classification_metrics(y, pred, len(classes))
    confusion = np.zeros((len(classes), len(classes)), dtype=int)
    for truth, guess in zip(y, pred):
        confusion[int(truth), int(guess)] += 1
    per_class = {}
    for c, name in enumerate(classes):
        tp = confusion[c, c]
        fn = confusion[c].sum() - tp
        fp = confusion[:, c].sum() - tp
        recall = tp / max(tp + fn, 1)
        precision = tp / max(tp + fp, 1)
        per_class[name] = {
            "support": int(confusion[c].sum()),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(2 * precision * recall / max(precision + recall, 1e-12)),
        }
    majority = int(np.bincount(y).argmax())
    # Exchangeability is at the same crude case-family level used by the CV,
    # not at individual word type.  Permute complete label vectors between
    # families of equal size.  This preserves global class margins, family
    # sizes, and the two mixed-family label patterns exactly.
    family_indices: dict[str, list[int]] = defaultdict(list)
    for i, word in enumerate(row_words):
        family_indices[strip_case(word)].append(i)
    family_size_strata: dict[int, list[list[int]]] = defaultdict(list)
    for family in sorted(family_indices):
        family_size_strata[len(family_indices[family])].append(
            sorted(family_indices[family])
        )

    rng_null = np.random.default_rng(SEED + 5)
    null_balanced = np.zeros(R_CONTEXT)
    null_accuracy = np.zeros(R_CONTEXT)
    for i in range(R_CONTEXT):
        yp = y.copy()
        for groups in family_size_strata.values():
            donor_order = rng_null.permutation(len(groups))
            donor_patterns = [y[groups[int(j)]].copy() for j in donor_order]
            for target, pattern in zip(groups, donor_patterns):
                yp[target] = pattern
        pp = crossval(yp)
        # Evaluation is against the correspondingly permuted family labels:
        # the null asks whether contexts carry more label information than
        # expected after preserving family dependence and class margins.
        a, b, _ = classification_metrics(yp, pp, len(classes))
        null_accuracy[i] = a
        null_balanced[i] = b
    out = {
        "records_excluding_ciep": len(clean_records),
        "labeled_types_freq_ge_2": len(row_words),
        "class_counts": dict(Counter(labels.values())),
        "context_dimensions": len(context_words),
        "nonzero_vectors": int((norms[:, 0] > 0).sum()),
        "family_blocked_folds": 5,
        "permutation_unit": "complete label vectors among equal-size case families",
        "permutations": R_CONTEXT,
        "accuracy": accuracy,
        "balanced_accuracy": balanced,
        "macro_f1": macro_f1,
        "classes": classes,
        "confusion": confusion.tolist(),
        "per_class": per_class,
        "majority_accuracy": float((y == majority).mean()),
        "null_accuracy_mean": float(null_accuracy.mean()),
        "null_balanced_mean": float(null_balanced.mean()),
        "p_balanced": float(((null_balanced >= balanced).sum() + 1) / (R_CONTEXT + 1)),
    }
    log("=== 3c. New approach pilot: context-only anchor transfer ===")
    log(
        f'excluding CIEP: {len(clean_records)} multiword records; '
        f'{len(row_words)} labeled freq>=2 types ({dict(Counter(labels.values()))})'
    )
    log(
        f'family-blocked PPMI centroid CV: acc={accuracy:.1%} '
        f'(majority={out["majority_accuracy"]:.1%}), balanced={balanced:.1%}, '
        f'macro-F1={macro_f1:.3f}; permutation balanced mean='
        f'{null_balanced.mean():.1%}, p={out["p_balanced"]:.4f}'
    )
    log()
    return out


# --- Morphology family -----------------------------------------------------
CANDIDATE_SUFFIXES = ["s", "l", "al", "ial", "us", "sa", "isa", "la", "na", "ce", "thi", "c", "m", "ei"]
REPORTED_CONTROL_PAIRS = [
    ("s", "al"), ("s", "l"), ("l", "al"), ("s", "ial"), ("l", "ial"),
    ("al", "sa"), ("s", "na"), ("l", "sa"), ("ial", "sa"), ("s", "isa"),
]


def exact_stratified_lower_p(U: set[str], A: set[str], B: set[str]) -> tuple[int, float, float]:
    strata: dict[tuple[str, int], dict[str, int]] = defaultdict(lambda: {"U": 0, "A": 0, "B": 0})
    for stem in U:
        key = (stem[-1], min(len(stem), 8))
        strata[key]["U"] += 1
        strata[key]["A"] += stem in A
        strata[key]["B"] += stem in B
    distribution = np.array([1.0])
    offset = 0
    expectation = 0.0
    for counts in strata.values():
        M, nA, nB = counts["U"], counts["A"], counts["B"]
        if not nA or not nB:
            continue
        lo = max(0, nA + nB - M)
        hi = min(nA, nB)
        ks = np.arange(lo, hi + 1)
        pmf = hypergeom.pmf(ks, M, nA, nB)
        pmf /= pmf.sum()
        distribution = np.convolve(distribution, pmf)
        offset += lo
        expectation += nA * nB / M
    observed = len(A & B)
    endpoint = observed - offset
    p = 0.0 if endpoint < 0 else float(distribution[: endpoint + 1].sum())
    return observed, expectation, min(max(p, 0.0), 1.0)


def suffix_screen(vocab: set[str], suffix: str) -> dict:
    k = len(suffix)
    eligible = [w for w in vocab if len(w) >= k + 2]
    N = len(eligible)
    K = sum(w[:-k] in vocab for w in eligible)
    ending = [w for w in eligible if w.endswith(suffix)]
    support = sum(w[:-k] in vocab for w in ending)
    p = float(hypergeom.sf(support - 1, N, K, len(ending))) if ending else 1.0
    return {
        "N": N, "K": K, "n_suffix": len(ending), "support": support,
        "expected": len(ending) * K / max(N, 1), "p": p,
    }


def longest_suffix_pair_analysis(vocab: set[str]) -> dict:
    """Retest the full suffix family after one longest-ending assignment."""
    longest_stems = {suffix: set() for suffix in CANDIDATE_SUFFIXES}
    longest_order = sorted(CANDIDATE_SUFFIXES, key=lambda s: (-len(s), s))
    for word in vocab:
        for suffix in longest_order:
            if word.endswith(suffix) and len(word) >= len(suffix) + 3:
                longest_stems[suffix].add(word[: -len(suffix)])
                break

    universe = set().union(*longest_stems.values())
    pairs = []
    for a, b in combinations(CANDIDATE_SUFFIXES, 2):
        observed, expected, p = exact_stratified_lower_p(
            universe, longest_stems[a], longest_stems[b]
        )
        pairs.append(
            {
                "pair": [a, b],
                "nested_suffixes": a.endswith(b) or b.endswith(a),
                "observed": observed,
                "expected": expected,
                "p_lower": p,
            }
        )

    nonnested = [row for row in pairs if not row["nested_suffixes"]]
    for family, family_name in ((pairs, "all"), (nonnested, "nonnested")):
        q_family = bh_adjust(np.array([row["p_lower"] for row in family]))
        for row, q_value in zip(family, q_family):
            row[f"q_bh_longest_{family_name}"] = float(q_value)
            row[f"p_bonf_longest_{family_name}"] = min(
                1.0, row["p_lower"] * len(family)
            )

    lookup = {tuple(row["pair"]): row for row in pairs}
    reported = []
    for a, b in REPORTED_CONTROL_PAIRS:
        key = (a, b) if (a, b) in lookup else (b, a)
        reported.append(lookup[key])

    return {
        "universe_stems": len(universe),
        "tested_pairs": len(pairs),
        "nonnested_pairs": len(nonnested),
        "all_pairs_bonferroni_lt_05": int(
            sum(row["p_bonf_longest_all"] < 0.05 for row in pairs)
        ),
        "nonnested_pairs_bonferroni_lt_05": int(
            sum(row["p_bonf_longest_nonnested"] < 0.05 for row in nonnested)
        ),
        "reported_ten_bonferroni_lt_05": int(
            sum(row["p_bonf_longest_all"] < 0.05 for row in reported)
        ),
        "reported_ten": reported,
    }


def morphology_audit(view: list[dict]) -> dict:
    vocab = {
        w
        for r in view
        for w in source_words(r)
        if "-" not in w and len(w) >= 2
    }
    stems = {
        suffix: {
            w[: -len(suffix)]
            for w in vocab
            if w.endswith(suffix) and len(w) >= len(suffix) + 3
        }
        for suffix in CANDIDATE_SUFFIXES
    }
    U = set().union(*stems.values())
    pairs = []
    for a, b in combinations(CANDIDATE_SUFFIXES, 2):
        observed, expected, p = exact_stratified_lower_p(U, stems[a], stems[b])
        pairs.append({"pair": [a, b], "observed": observed, "expected": expected, "p_lower": p})
    pvals = np.array([row["p_lower"] for row in pairs])
    qvals = bh_adjust(pvals)
    for row, q in zip(pairs, qvals):
        row["q_bh_91"] = float(q)
        row["p_bonf_91"] = min(1.0, row["p_lower"] * len(pairs))
        row["p_bonf_182"] = min(1.0, row["p_lower"] * 2 * len(pairs))
    pair_lookup = {tuple(row["pair"]): row for row in pairs}
    reported = []
    for a, b in REPORTED_CONTROL_PAIRS:
        key = (a, b) if (a, b) in pair_lookup else (b, a)
        reported.append(pair_lookup[key])

    # Ablation requested in the audit: the production code lets every word
    # match every compatible ending.  Here each word is assigned only to its
    # longest candidate ending.  This removes e.g. X-ial from the -al/-l
    # pools and X-isa from the -sa pool.
    longest_match = longest_suffix_pair_analysis(vocab)

    # Sensitivity analysis: a form enters this vocabulary only if it has at
    # least one occurrence without unresolved record-level OCR and without an
    # epistemic token flag.  This keeps editorially uncertain readings from
    # carrying the headline complementarity signal.
    excluded_token_flags = {"damaged", "uncertain", "restored", "emended"}
    clean_tokens = [
        token
        for record in view
        if "ocr?" not in record["flags"]
        for token in record["toks"]
        if token["kind"] == "W"
        and not (set(token["flags"]) & excluded_token_flags)
        and "-" not in token["form"]
        and len(token["form"]) >= 2
    ]
    clean_vocab = {token["form"] for token in clean_tokens}
    clean_reading = longest_suffix_pair_analysis(clean_vocab)
    clean_reading.update(
        {
            "definition": (
                "form has >=1 token occurrence outside record flag ocr? and "
                "without token flags damaged/uncertain/restored/emended"
            ),
            "vocab_types": len(clean_vocab),
            "eligible_token_occurrences": len(clean_tokens),
            "excluded_record_flag": "ocr?",
            "excluded_token_flags": sorted(excluded_token_flags),
        }
    )

    source_replication = {}
    source_groups = {
        "all": view,
        "ETP": [r for r in view if r["src"] == "ETP"],
        "CIEP": [r for r in view if r["src"] == "CIEP"],
        "CIEW": [r for r in view if r["src"] == "CIEW"],
        "CIEW-CIE": [r for r in view if r["src"] == "CIEW-CIE"],
        "non_ETP": [r for r in view if r["src"] != "ETP"],
    }
    for name, recs in source_groups.items():
        vv = {
            w
            for r in recs
            for w in source_words(r)
            if "-" not in w and len(w) >= 2
        }
        ial = {w[:-3] for w in vv if w.endswith("ial") and len(w) >= 6}
        ei = {w[:-2] for w in vv if w.endswith("ei") and len(w) >= 5}
        source_replication[name] = {
            "vocab": len(vv), "ial_stems": len(ial), "ei_stems": len(ei),
            "overlap": len(ial & ei), "examples": sorted(ial & ei)[:30],
            "suffix_c": suffix_screen(vv, "c"),
        }

    out = {
        "orthography": "normalized source forms (token.form), not lossy ASCII projection",
        "universe_stems": len(U),
        "tested_pairs": len(pairs),
        "lower_tail_bonferroni_91_lt_05": int(sum(r["p_bonf_91"] < 0.05 for r in pairs)),
        "two_direction_bonferroni_182_lt_05": int(sum(r["p_bonf_182"] < 0.05 for r in pairs)),
        "bh_91_lt_05": int(sum(r["q_bh_91"] < 0.05 for r in pairs)),
        "top_lower_tail": sorted(pairs, key=lambda x: x["p_lower"])[:20],
        "reported_ten_retested_in_full_family": reported,
        "longest_match": longest_match,
        "clean_reading_longest_match_sensitivity": clean_reading,
        "ial_ei_and_suffix_c_by_source": source_replication,
    }
    log("=== 4. Morphology: selection family and source replication ===")
    log(
        f'all {len(pairs)} candidate pairs retested with exact final+length-stratified null: '
        f'{out["lower_tail_bonferroni_91_lt_05"]} survive lower-tail Bonferroni, '
        f'{out["two_direction_bonferroni_182_lt_05"]} survive the two-direction family'
    )
    for row in reported:
        log(
            f'  -{row["pair"][0]}/-{row["pair"][1]} obs={row["observed"]} '
            f'E={row["expected"]:.1f} p={row["p_lower"]:.3g} '
            f'pBonf182={row["p_bonf_182"]:.3g}'
        )
    log(
        f'longest-match ablation: {longest_match["all_pairs_bonferroni_lt_05"]}/91 '
        f'pairs survive Bonferroni; after excluding nested endings, '
        f'{longest_match["nonnested_pairs_bonferroni_lt_05"]}/'
        f'{longest_match["nonnested_pairs"]}'
    )
    for row in longest_match["reported_ten"]:
        log(
            f'  longest -{row["pair"][0]}/-{row["pair"][1]} '
            f'nested={row["nested_suffixes"]} obs={row["observed"]} '
            f'E={row["expected"]:.1f} p={row["p_lower"]:.3g} '
            f'pBonf91={row["p_bonf_longest_all"]:.3g}'
        )
    log(
        "clean-reading sensitivity (exclude ocr? records and "
        "damaged/uncertain/restored/emended tokens): "
        f'{clean_reading["reported_ten_bonferroni_lt_05"]}/10 reported pairs, '
        f'{clean_reading["all_pairs_bonferroni_lt_05"]}/91 all pairs survive Bonferroni'
    )
    for row in clean_reading["reported_ten"]:
        log(
            f'  clean longest -{row["pair"][0]}/-{row["pair"][1]} '
            f'obs={row["observed"]} E={row["expected"]:.1f} '
            f'p={row["p_lower"]:.3g} pBonf91={row["p_bonf_longest_all"]:.3g}'
        )
    for name in ("ETP", "CIEP", "CIEW", "CIEW-CIE", "all"):
        row = source_replication[name]
        c = row["suffix_c"]
        log(
            f'  {name}: -ial/-ei exact shared stems {row["overlap"]} '
            f'({row["ial_stems"]} x {row["ei_stems"]}); -c support '
            f'{c["support"]}/{c["n_suffix"]}, E={c["expected"]:.1f}, p={c["p"]:.3g}'
        )
    log()
    return out


# --- Prosopography and dating cluster nulls -------------------------------
CASE_ENDS = ("isa", "ial", "al", "us", "sa", "s", "l")
NOT_NAMES = set(
    "mi mini mine clan clens sec sech puia ati apa lupu lupuce svalce avils avil ril "
    "turce turuce muluvanice mulvanice zinace zilath zilc suthi thui cver tular ame "
    "amce itun ita ica eca ca cn cen etnam vacl fler naper tiur".split()
)


def name_stem(w: str) -> str:
    for ending in CASE_ENDS:
        if w.endswith(ending) and len(w) - len(ending) >= 3:
            return w[: -len(ending)]
    return w


def name_types_for_audit(view: list[dict]) -> set[str]:
    names = set()
    for r in view:
        ws = words(r)
        if r["trs"] and len(ws) == 1 and len(ws[0]) >= 3 and "-" not in ws[0]:
            if gloss_class(" ".join(r["trs"]).lower()).startswith("NAME"):
                names.add(ws[0])
    with open(os.path.join("data", "ETP_POS.csv"), encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            w = to_ascii_word(row.get("Etruscan"))
            if len(w) >= 3 and ((row.get("masc") or "").strip() == "1" or (row.get("fem") or "").strip() == "1"):
                if (row.get("theo") or "").strip() != "1":
                    names.add(w)
    return names - NOT_NAMES


def mutual_information(stems: list[str], regions: list[str]) -> float:
    svals = sorted(set(stems))
    rvals = sorted(set(regions))
    si = {s: i for i, s in enumerate(svals)}
    ri = {r: i for i, r in enumerate(rvals)}
    M = np.zeros((len(svals), len(rvals)), dtype=float)
    for s, r in zip(stems, regions):
        M[si[s], ri[r]] += 1
    P = M / M.sum()
    pr = P.sum(axis=1, keepdims=True)
    pc = P.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = P * np.log(P / (pr @ pc))
    return float(np.nansum(terms))


def cluster_null_audit(view: list[dict]) -> dict:
    names = name_types_for_audit(view)
    cluster_rows: dict[str, dict] = {}
    for r in view:
        if not r["region"]:
            continue
        cid = artifact_id(r)
        # Inconsistent xref clusters are conservatively split by region.
        key = cid + "|" + r["region"]
        d = cluster_rows.setdefault(key, {"region": r["region"], "source": r["src"].split(":")[0], "stems": []})
        d["stems"].extend(name_stem(w) for w in words(r) if w in names)
    cluster_rows = {k: v for k, v in cluster_rows.items() if v["stems"]}
    keys = sorted(cluster_rows)
    stems = [s for k in keys for s in cluster_rows[k]["stems"]]
    regions = [cluster_rows[k]["region"] for k in keys for _ in cluster_rows[k]["stems"]]
    observed = mutual_information(stems, regions)
    rng = np.random.default_rng(SEED + 2)
    global_null = np.zeros(R_CLUSTER)
    strat_null = np.zeros(R_CLUSTER)
    original_regions = np.array([cluster_rows[k]["region"] for k in keys], dtype=object)
    sources = np.array([cluster_rows[k]["source"] for k in keys], dtype=object)

    def expand(region_by_cluster: np.ndarray) -> list[str]:
        return [region_by_cluster[i] for i, k in enumerate(keys) for _ in cluster_rows[k]["stems"]]

    strata = [np.where(sources == src)[0] for src in sorted(set(sources))]
    for i in range(R_CLUSTER):
        perm = original_regions[rng.permutation(len(keys))]
        global_null[i] = mutual_information(stems, expand(perm))
        perm2 = original_regions.copy()
        for idx in strata:
            perm2[idx] = perm2[idx][rng.permutation(len(idx))]
        strat_null[i] = mutual_information(stems, expand(perm2))
    out = {
        "name_occurrences": len(stems),
        "artifact_proxy_clusters": len(keys),
        "observed_mi": observed,
        "cluster_global_null_mean": float(global_null.mean()),
        "cluster_global_p": float(((global_null >= observed).sum() + 1) / (R_CLUSTER + 1)),
        "cluster_source_stratified_null_mean": float(strat_null.mean()),
        "cluster_source_stratified_p": float(((strat_null >= observed).sum() + 1) / (R_CLUSTER + 1)),
    }
    log("=== 5. Prosopography: artifact-cluster permutation ===")
    log(
        f'MI={observed:.4f} on {len(stems)} name occurrences in {len(keys)} clusters; '
        f'cluster-null mean={global_null.mean():.4f}, p={out["cluster_global_p"]:.4f}; '
        f'source-stratified cluster-null mean={strat_null.mean():.4f}, '
        f'p={out["cluster_source_stratified_p"]:.4f}'
    )
    log()
    return out


def century_bin(year_bce: float) -> str:
    if year_bce >= 550:
        return "before_550"
    if year_bce >= 300:
        return "550_to_300"
    return "after_300"


def stratigraphy_record_null_audit(view: list[dict]) -> dict:
    dated = [r for r in view if r["y_from"] is not None]
    records = []
    total_stems = Counter()
    for r in dated:
        stems = [
            name_stem(w) for w in words(r)
            if "-" not in w and len(w) >= 4
        ]
        total_stems.update(stems)
        records.append(
            {
                "stems": stems,
                "early": century_bin(float(r["y_from"])),
                "mid": century_bin(
                    (float(r["y_from"]) + float(r["y_to"])) / 2
                    if r["y_to"] is not None else float(r["y_from"])
                ),
                "from": float(r["y_from"]),
                "to": float(r["y_to"]) if r["y_to"] is not None else float(r["y_from"]),
                "geo": r["region"] or r["city"] or "?",
            }
        )
    tested = {s for s, n in total_stems.items() if n >= 3}
    stem_occ = [s for rec in records for s in rec["stems"] if s in tested]

    def expand(labels: list[str] | np.ndarray) -> list[str]:
        return [str(labels[i]) for i, rec in enumerate(records) for s in rec["stems"] if s in tested]

    early = np.array([rec["early"] for rec in records], dtype=object)
    mid = np.array([rec["mid"] for rec in records], dtype=object)
    observed_early = mutual_information(stem_occ, expand(early))
    observed_mid = mutual_information(stem_occ, expand(mid))
    rng = np.random.default_rng(SEED + 6)
    record_null = np.zeros(R_CLUSTER)
    geo_null = np.zeros(R_CLUSTER)
    geos = np.array([rec["geo"] for rec in records], dtype=object)
    geo_strata = [np.where(geos == g)[0] for g in sorted(set(geos))]
    for i in range(R_CLUSTER):
        record_null[i] = mutual_information(stem_occ, expand(early[rng.permutation(len(early))]))
        perm = early.copy()
        for idx in geo_strata:
            perm[idx] = perm[idx][rng.permutation(len(idx))]
        geo_null[i] = mutual_information(stem_occ, expand(perm))

    imputed = np.zeros(R_CLUSTER)
    for i in range(R_CLUSTER):
        sampled_bins = []
        for rec in records:
            lo, hi = sorted((rec["to"], rec["from"]))
            sampled_bins.append(century_bin(float(rng.uniform(lo, hi))) if hi > lo else century_bin(lo))
        imputed[i] = mutual_information(stem_occ, expand(np.array(sampled_bins, dtype=object)))

    spans = 0
    for rec in records:
        lo, hi = sorted((rec["to"], rec["from"]))
        possible = {century_bin(lo), century_bin(hi)}
        if len(possible) > 1:
            spans += 1
    out = {
        "dated_records": len(records),
        "tested_stems": len(tested),
        "tested_occurrences": len(stem_occ),
        "intervals_crossing_bin_boundary": spans,
        "early_vs_mid_bin_changes": int((early != mid).sum()),
        "mi_early_bound": observed_early,
        "mi_midpoint": observed_mid,
        "record_level_null_mean": float(record_null.mean()),
        "record_level_p": float(((record_null >= observed_early).sum() + 1) / (R_CLUSTER + 1)),
        "geography_stratified_record_null_mean": float(geo_null.mean()),
        "geography_stratified_record_p": float(((geo_null >= observed_early).sum() + 1) / (R_CLUSTER + 1)),
        "date_interval_imputation_mi_median": float(np.median(imputed)),
        "date_interval_imputation_mi_95": [float(np.quantile(imputed, 0.025)), float(np.quantile(imputed, 0.975))],
    }
    log("=== 5b. Stratigraphy: record-level null and date intervals ===")
    log(
        f'{len(records)} dated records; {spans} intervals cross a bin boundary; '
        f'{int((early != mid).sum())} assignments change using midpoint'
    )
    log(
        f'MI early={observed_early:.4f}, midpoint={observed_mid:.4f}; '
        f'record-null mean={record_null.mean():.4f}, p={out["record_level_p"]:.4f}; '
        f'geography-stratified null mean={geo_null.mean():.4f}, '
        f'p={out["geography_stratified_record_p"]:.4f}'
    )
    log(
        f'interval-imputed MI median={np.median(imputed):.4f}, '
        f'95% [{np.quantile(imputed, .025):.4f}, {np.quantile(imputed, .975):.4f}]'
    )
    log()
    return out


# --- Lemnian classifier leakage -------------------------------------------
def language_tokens(corpus: dict, language: str, source: str | None = None) -> list[str]:
    out = []
    for r in corpus["records"]:
        if r["lang"] == language and r["kind"] == "text" and "forgery?" not in r["flags"] and r.get("variant_of") is None and (source is None or r["src"] == source):
            out.extend(t["ascii"] for t in r["toks"] if t["kind"] == "W" and "-" not in t["ascii"] and len(t["ascii"]) >= 3)
    return out


def language_features(w: str) -> set[str]:
    out = set()
    for k in (1, 2, 3):
        if len(w) > k:
            out.add(f"s{k}:{w[-k:]}")
    for k in (1, 2):
        if len(w) > k:
            out.add(f"p{k}:{w[:k]}")
    return out


def lemnos_audit(corpus: dict) -> dict:
    pools = {lg: language_tokens(corpus, lg) for lg in ("etr", "lat", "umb")}
    # The CIEP 15999 rows are partial readings of the same monument.
    lemn = language_tokens(corpus, "lemn", "SUPP:lemnos_wikipedia.csv")
    rng = np.random.default_rng(42)
    original_leakage = {}
    for lg in sorted(pools):
        p = pools[lg]
        idx = rng.permutation(len(p))
        train = [p[i] for i in idx[:120]]
        hold = [p[i] for i in idx[120:]]
        train_types = set(train)
        original_leakage[lg] = {
            "train_tokens": len(train),
            "holdout_tokens": len(hold),
            "holdout_tokens_exact_in_train": int(sum(w in train_types for w in hold)),
            "holdout_exact_leak_share": float(np.mean([w in train_types for w in hold])) if hold else 0.0,
            "holdout_unique_types_exact_in_train": len(set(hold) & train_types),
        }

    rng2 = np.random.default_rng(SEED + 3)
    unique = {lg: sorted(set(values)) for lg, values in pools.items()}
    control_units = {}
    for lg in ("lat", "umb"):
        rr = [
            r for r in corpus["records"]
            if r["lang"] == lg and r["kind"] == "text"
            and "forgery?" not in r["flags"] and r.get("variant_of") is None
        ]
        control_units[lg] = {
            "records": len(rr),
            "source_eids": len({(r["src"], r["eid"]) for r in rr}),
            "artifact_proxy_clusters": len({artifact_id(r) for r in rr}),
        }
    n_train = min(120, min(len(v) for v in unique.values()) // 2)
    model_names = ("present_only_project", "bernoulli", "multinomial")
    correct = Counter()
    total = Counter()
    lemn_predictions = {name: Counter() for name in model_names}
    repetitions = 100
    bags_per_language = 20
    for _ in range(repetitions):
        train = {}
        hold = {}
        for lg in sorted(unique):
            perm = rng2.permutation(len(unique[lg]))
            train[lg] = [unique[lg][i] for i in perm[:n_train]]
            hold[lg] = [unique[lg][i] for i in perm[n_train:]]
        feature_index = {}
        for lg in train:
            for w in train[lg]:
                for f in language_features(w):
                    feature_index.setdefault(f, len(feature_index))

        train_counts = {
            lg: Counter(f for w in train[lg] for f in language_features(w))
            for lg in train
        }
        V = len(feature_index)

        def log_likelihood(bag: list[str], model: str) -> dict[str, float]:
            scores = {}
            bag_features = [
                {f for f in language_features(w) if f in feature_index}
                for w in bag
            ]
            for lg in train:
                counts = train_counts[lg]
                if model == "present_only_project":
                    score = sum(
                        math.log((counts.get(f, 0) + 1) / (len(train[lg]) + 2))
                        for fs in bag_features for f in fs
                    )
                elif model == "bernoulli":
                    probabilities = np.array(
                        [(counts.get(f, 0) + 1) / (len(train[lg]) + 2)
                         for f, _ in sorted(feature_index.items(), key=lambda x: x[1])]
                    )
                    log0 = np.log1p(-probabilities)
                    delta = np.log(probabilities) - log0
                    score = 0.0
                    base = float(log0.sum())
                    for fs in bag_features:
                        score += base + sum(delta[feature_index[f]] for f in fs)
                else:  # proper multinomial NB over observed affix features
                    denominator = sum(counts.values()) + V
                    score = sum(
                        math.log((counts.get(f, 0) + 1) / denominator)
                        for fs in bag_features for f in fs
                    )
                scores[lg] = score / max(len(bag), 1)
            return scores

        for lg in sorted(hold):
            for _ in range(bags_per_language):
                bag = [hold[lg][i] for i in rng2.integers(0, len(hold[lg]), len(lemn))]
                for model in model_names:
                    scores = log_likelihood(bag, model)
                    pred = max(scores, key=scores.get)
                    correct[model] += pred == lg
                    total[model] += 1
        for model in model_names:
            ll_lemn = log_likelihood(lemn, model)
            lemn_predictions[model][max(ll_lemn, key=ll_lemn.get)] += 1

    out = {
        "pool_tokens": {lg: len(v) for lg, v in pools.items()},
        "pool_unique_types": {lg: len(v) for lg, v in unique.items()},
        "control_units": control_units,
        "lemnian_tokens": len(lemn),
        "lemnian_unique_types": len(set(lemn)),
        "original_token_split_exact_leakage": original_leakage,
        "type_disjoint_train_types_per_language": n_train,
        "type_disjoint_models": {
            model: {
                "calibration_bag_accuracy": correct[model] / total[model],
                "calibration_bags": total[model],
                "lemnian_prediction_across_100_splits": dict(lemn_predictions[model]),
            }
            for model in model_names
        },
    }
    log("=== 6. Lemnian classifier: exact-type leakage and type-disjoint rerun ===")
    for lg in sorted(original_leakage):
        row = original_leakage[lg]
        log(
            f'  original {lg} holdout exact-token leakage: '
            f'{row["holdout_tokens_exact_in_train"]}/{row["holdout_tokens"]} '
            f'({row["holdout_exact_leak_share"]:.1%})'
        )
    for model in model_names:
        log(
            f'  type-disjoint {model}: {correct[model]}/{total[model]} bags correct '
            f'({correct[model]/total[model]:.1%}); Lemnian over 100 splits: '
            f'{dict(lemn_predictions[model])}'
        )
    log()
    return out


def main() -> None:
    os.makedirs("results", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    corpus_path = os.path.join("data", "etr_corpus.pkl")
    declared_path = os.path.join("data", "etr_corpus.sha256")
    actual_corpus_sha = sha256_file(corpus_path)
    declared_line = open(declared_path, encoding="utf-8").read().strip()
    declared_corpus_sha = declared_line.split()[0]
    assert actual_corpus_sha == declared_corpus_sha, (
        "etr_corpus.pkl does not match data/etr_corpus.sha256"
    )
    with open(corpus_path, "rb") as handle:
        corpus = pickle.load(handle)
    assert corpus["meta"].get("freeze_version") == "0.9"
    view = canonical_view(corpus)
    input_paths = {
        "corpus": corpus_path,
        "etp_pos": os.path.join("data", "ETP_POS.csv"),
        "concept_lexicon": os.path.join("data", "concepts", "concept_lexicon.csv"),
        "semantic_hypotheses_v2": os.path.join("results", "semantic_hypotheses_v2.csv"),
        "concept_fog_v1": os.path.join("results", "concept_fog_v1.csv"),
        "script": os.path.abspath(__file__),
    }
    result = {
        "audit_version": "sol-2026-07-10",
        "seed": SEED,
        "corpus_sha256_declared_file": declared_line,
        "meta": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "permutations": {
                "concept": R_CONCEPT,
                "semantic": R_SEMANTIC,
                "cluster": R_CLUSTER,
                "context": R_CONTEXT,
            },
            "input_sha256": {
                name: sha256_file(path) for name, path in input_paths.items()
            },
            "declared_corpus_sha256": declared_corpus_sha,
        },
        "corpus": corpus_audit(corpus, view),
        "concept_convergence": concept_audit(),
        "semantic_cross_provenance": semantic_audit(view),
        "operator_validation_by_provenance": operator_validation_audit(view),
        "distributional_anchor_pilot": distributional_anchor_audit(view),
        "morphology": morphology_audit(view),
        "prosopography_cluster_null": cluster_null_audit(view),
        "stratigraphy_record_null": stratigraphy_record_null_audit(view),
        "lemnos_type_leakage": lemnos_audit(corpus),
    }
    with open(OUT_JSON, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    with open(OUT_LOG, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(LOG).rstrip() + "\n")
    print(f"written: {OUT_JSON}")
    print(f"written: {OUT_LOG}")


if __name__ == "__main__":
    main()
