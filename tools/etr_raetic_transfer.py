# -*- coding: utf-8 -*-
"""External Raetic -> Etruscan case-transfer experiment.

The model is deliberately small and interpretable: a categorical Naive Bayes
classifier trained *only* on clean, explicitly Raetic TIR word entries and
their expert case labels.  Its only features are final 1--4-grams.  It is then
evaluated, without refitting, on clean, attested, non-suffix ETP_POS types that
occur in the frozen Etruscan corpus.

This is a transfer/validation experiment, not independent proof of Tyrsenian
relationship: TIR's philological analyses themselves use Etruscan comparison.
It asks the narrower, measurable question whether the published Raetic case
system is predictive enough to rank Etruscan case candidates.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import pickle
import sys
import unicodedata
from collections import Counter, defaultdict

import numpy as np


sys.stdout.reconfigure(encoding="utf-8")

SEED = 42
R = 10_000
ALPHA = 0.5
CLASSES = ("GEN", "NOM", "PERT")
CLASS_INDEX = {name: i for i, name in enumerate(CLASSES)}
CASE_OF_TIR = {
    "genitive": "GEN",
    "nominative": "NOM",
    "pertinentive": "PERT",
}
OUT_LOG = os.path.join("logs", "etr_raetic_transfer.log")
OUT_JSON = os.path.join("results", "raetic_transfer_validation.json")
OUT_CSV = os.path.join("results", "raetic_transfer_candidates.csv")
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


def clean_surface(value: str) -> bool:
    """Accept a full lexical form, not an editorial reconstruction/template."""
    value = unicodedata.normalize("NFC", (value or "").strip())
    if not value:
        return False
    # Apostrophes are transcriptional parts of a sibilant in the source data.
    return all(ch.isalpha() or ch in "'’ʼ" for ch in value)


def fold_word(value: str, mode: str) -> str:
    value = unicodedata.normalize("NFC", (value or "").strip().lower())
    value = value.replace("’", "'").replace("ʼ", "'").replace("'", "")
    if mode == "strict":
        return "".join(ch for ch in value if ch.isalpha())
    if mode == "minimal":
        table = str.maketrans(
            {
                "c": "k",
                "q": "k",
                "ś": "s",
                "š": "s",
                "σ": "s",
                "ς": "s",
                "ê": "e",
            }
        )
        return "".join(ch for ch in value.translate(table) if ch.isalpha())
    raise ValueError(mode)


def suffixes(word: str, max_k: int = 4) -> list[tuple[int, str]]:
    return [(k, word[-k:]) for k in range(1, max_k + 1) if len(word) > k]


def load_tir(
    mode: str, *, checklevel_zero_only: bool = False
) -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    path = os.path.join("data", "external", "tir", "tir_words.csv")
    with open(path, encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            label = CASE_OF_TIR.get(row["case"])
            title = row["title"]
            if row["language"] != "Raetic" or not label or not clean_surface(title):
                continue
            # TIR defines checklevel 0 as "nothing to correct"; higher values
            # are retained in the primary analysis and excluded in a quality
            # sensitivity analysis below.
            if checklevel_zero_only and row.get("checklevel") != "0":
                continue
            word = fold_word(title, mode)
            if len(word) < 3:
                continue
            lemma = fold_word(row.get("lemma") or title, mode)
            rows.append((word, label, lemma))
    return rows


def corpus_view(corpus: dict) -> list[dict]:
    return [
        row
        for row in corpus["records"]
        if row["lang"] == "etr"
        and row["kind"] == "text"
        and "forgery?" not in row["flags"]
        and row.get("variant_of") is None
    ]


def load_etp_gold(mode: str, corpus: dict) -> tuple[list[tuple[str, str]], dict]:
    all_corpus_types = {
        fold_word(token["form"], mode)
        for row in corpus_view(corpus)
        for token in row["toks"]
        if token["kind"] == "W" and clean_surface(token["form"])
    }
    corpus_types = {
        fold_word(token["form"], mode)
        for row in corpus_view(corpus)
        if "ocr?" not in row["flags"]
        for token in row["toks"]
        if token["kind"] == "W"
        and not token["flags"]
        and clean_surface(token["form"])
    }
    labels: dict[str, set[str]] = defaultdict(set)
    with open(os.path.join("data", "ETP_POS.csv"), encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("Is suffix") == "True" or row.get("Is inferred") == "True":
                continue
            surface = row.get("Etruscan") or ""
            if not clean_surface(surface):
                continue
            found: list[str] = []
            if row.get("nom") == "1":
                found.append("NOM")
            if row.get("gen") == "True":
                found.append("GEN")
            if row.get("pert") == "True":
                found.append("PERT")
            word = fold_word(surface, mode)
            if len(found) == 1 and len(word) >= 3 and word in all_corpus_types:
                labels[word].add(found[0])
    conflicts = sorted(word for word, values in labels.items() if len(values) > 1)
    gold = sorted(
        (word, next(iter(values)))
        for word, values in labels.items()
        if len(values) == 1 and word in corpus_types
    )
    only_flagged = sorted(
        word
        for word, values in labels.items()
        if len(values) == 1 and word in all_corpus_types and word not in corpus_types
    )
    return gold, {
        "conflicting_normalized_types": conflicts,
        "excluded_gold_types_without_clean_token_attestation": only_flagged,
    }


class SuffixNB:
    def __init__(self, max_k: int = 4, alpha: float = ALPHA):
        self.max_k = max_k
        self.alpha = alpha
        self.vocab: dict[int, list[str]] = {}
        self.logp: dict[int, np.ndarray] = {}
        self.support: dict[int, dict[str, Counter]] = {}

    def fit(self, rows: list[tuple[str, str, str]]) -> "SuffixNB":
        for k in range(1, self.max_k + 1):
            eligible = [(word[-k:], label) for word, label, _ in rows if len(word) > k]
            vocab = sorted({ending for ending, _ in eligible})
            index = {ending: i for i, ending in enumerate(vocab)}
            counts = np.zeros((len(CLASSES), len(vocab)), dtype=float)
            totals = np.zeros(len(CLASSES), dtype=float)
            for ending, label in eligible:
                c = CLASS_INDEX[label]
                counts[c, index[ending]] += 1
                totals[c] += 1
            self.vocab[k] = vocab
            self.support[k] = {
                ending: Counter(label for value, label in eligible if value == ending)
                for ending in vocab
            }
            self.logp[k] = np.log(
                (counts + self.alpha)
                / (totals[:, None] + self.alpha * max(len(vocab), 1))
            )
        return self

    def scores(self, words: list[str]) -> np.ndarray:
        scores = np.zeros((len(words), len(CLASSES)), dtype=float)
        for k in range(1, self.max_k + 1):
            index = {ending: i for i, ending in enumerate(self.vocab[k])}
            for i, word in enumerate(words):
                if len(word) <= k:
                    continue
                j = index.get(word[-k:])
                if j is not None:
                    scores[i] += self.logp[k][:, j]
        return scores

    def evidence_mask(self, words: list[str], min_k: int = 1) -> np.ndarray:
        """Whether a word has at least one suffix observed in training."""
        known = {k: set(values) for k, values in self.vocab.items()}
        return np.array(
            [
                any(
                    len(word) > k and word[-k:] in known[k]
                    for k in range(min_k, self.max_k + 1)
                )
                for word in words
            ],
            dtype=bool,
        )

    def longest_scores(
        self, words: list[str], min_k: int = 2
    ) -> tuple[np.ndarray, list[str]]:
        """Score with one longest known suffix, avoiding nested double-count."""
        scores = np.zeros((len(words), len(CLASSES)), dtype=float)
        selected = [""] * len(words)
        indices = {
            k: {ending: i for i, ending in enumerate(self.vocab[k])}
            for k in range(min_k, self.max_k + 1)
        }
        for i, word in enumerate(words):
            for k in range(self.max_k, min_k - 1, -1):
                if len(word) <= k:
                    continue
                ending = word[-k:]
                j = indices[k].get(ending)
                if j is not None:
                    scores[i] = self.logp[k][:, j]
                    selected[i] = f"{k}:{ending}"
                    break
        return scores, selected


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    if len(y_true) == 0:
        raise ValueError("metrics require at least one evaluated item")
    confusion = np.array(
        [
            [int(((y_true == a) & (y_pred == b)).sum()) for b in range(len(CLASSES))]
            for a in range(len(CLASSES))
        ]
    )
    recalls = []
    f1s = []
    per_class = {}
    for c, name in enumerate(CLASSES):
        tp = int(confusion[c, c])
        fn = int(confusion[c].sum() - tp)
        fp = int(confusion[:, c].sum() - tp)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        precision_ci = wilson_interval(tp, tp + fp)
        recalls.append(recall)
        f1s.append(f1)
        per_class[name] = {
            "n": int((y_true == c).sum()),
            "n_true": int((y_true == c).sum()),
            "n_predicted": int((y_pred == c).sum()),
            "true_positive": tp,
            "precision": precision,
            "precision_wilson_95ci": precision_ci,
            "recall": recall,
            "f1": f1,
        }
    return {
        "accuracy": float((y_true == y_pred).mean()),
        "balanced_accuracy": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1s)),
        "confusion_rows_true_cols_pred": confusion.tolist(),
        "per_class": per_class,
    }


def wilson_interval(successes: int, total: int) -> list[float | None]:
    if total == 0:
        return [None, None]
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    radius = (
        z
        * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
        / denominator
    )
    return [max(0.0, centre - radius), min(1.0, centre + radius)]


def metrics_with_abstention(
    y_true: np.ndarray, y_pred: np.ndarray, evaluated: np.ndarray
) -> dict:
    result = metrics(y_true[evaluated], y_pred[evaluated])
    n_total = len(y_true)
    n_evaluated = int(evaluated.sum())
    result.update(
        {
            "n_total": n_total,
            "n_evaluated": n_evaluated,
            "n_abstained": n_total - n_evaluated,
            "coverage": n_evaluated / n_total,
            "abstention_rate": 1 - n_evaluated / n_total,
        }
    )
    return result


CASE_SUFFIXES = ("ial", "ale", "al", "la", "si", "le", "s", "l")


def family_key(word: str) -> str:
    for ending in CASE_SUFFIXES:
        if word.endswith(ending) and len(word) >= len(ending) + 3:
            return word[: -len(ending)]
    return word


def cluster_bootstrap(
    y_true: np.ndarray, y_pred: np.ndarray, words: list[str], rng: np.random.Generator
) -> tuple[float, float]:
    groups: dict[str, list[int]] = defaultdict(list)
    for i, word in enumerate(words):
        groups[family_key(word)].append(i)
    keys = sorted(groups)
    values = []
    for _ in range(5_000):
        sampled = rng.integers(0, len(keys), len(keys))
        idx = np.array([i for g in sampled for i in groups[keys[int(g)]]], dtype=int)
        if all((y_true[idx] == c).any() for c in range(len(CLASSES))):
            values.append(metrics(y_true[idx], y_pred[idx])["balanced_accuracy"])
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def equal_size_group_strata(keys: list[str]) -> list[list[np.ndarray]]:
    """Group row-index blocks by size so vectors can be permuted intact."""
    groups: dict[str, list[int]] = defaultdict(list)
    for i, key in enumerate(keys):
        groups[key].append(i)
    by_size: dict[int, list[np.ndarray]] = defaultdict(list)
    for idx in groups.values():
        by_size[len(idx)].append(np.asarray(idx, dtype=int))
    return [by_size[size] for size in sorted(by_size)]


def block_permute_labels(
    labels: np.ndarray,
    strata: list[list[np.ndarray]],
    rng: np.random.Generator,
) -> np.ndarray:
    """Permute whole label vectors among equal-size blocks.

    The operation preserves global class marginals exactly as well as the
    within-block label vectors.  Only their association with word/lemma blocks
    is randomized.
    """
    shuffled = labels.copy()
    for blocks in strata:
        source_order = rng.permutation(len(blocks))
        for destination, source_i in zip(blocks, source_order):
            shuffled[destination] = labels[blocks[int(source_i)]]
    return shuffled


def family_block_label_null(
    y_true: np.ndarray, y_pred: np.ndarray, words: list[str], rng: np.random.Generator
) -> tuple[float, float, float]:
    """Marginal-preserving permutation of label vectors among stem families."""
    strata = equal_size_group_strata([family_key(word) for word in words])
    observed = metrics(y_true, y_pred)["balanced_accuracy"]
    sims = np.zeros(R)
    for r_i in range(R):
        shuffled = block_permute_labels(y_true, strata, rng)
        sims[r_i] = metrics(shuffled, y_pred)["balanced_accuracy"]
    p = float(((sims >= observed).sum() + 1) / (R + 1))
    return p, float(sims.mean()), float(np.quantile(sims, 0.95))


def training_label_null(
    train: list[tuple[str, str, str]],
    test_words: list[str],
    y_true: np.ndarray,
    observed: float,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    labels = np.array([CLASS_INDEX[label] for _, label, _ in train], dtype=int)
    lemma_strata = equal_size_group_strata([lemma or word for word, _, lemma in train])
    train_mats: dict[int, np.ndarray] = {}
    test_mats: dict[int, np.ndarray] = {}
    eligible: dict[int, np.ndarray] = {}
    for k in range(1, 5):
        vocab = sorted({word[-k:] for word, _, _ in train if len(word) > k})
        index = {ending: i for i, ending in enumerate(vocab)}
        train_mat = np.zeros((len(train), len(vocab)), dtype=np.int8)
        test_mat = np.zeros((len(test_words), len(vocab)), dtype=np.int8)
        eligible[k] = np.array([len(word) > k for word, _, _ in train])
        for i, (word, _, _) in enumerate(train):
            if len(word) > k:
                train_mat[i, index[word[-k:]]] = 1
        for i, word in enumerate(test_words):
            if len(word) > k and word[-k:] in index:
                test_mat[i, index[word[-k:]]] = 1
        train_mats[k] = train_mat
        test_mats[k] = test_mat
    sims = np.zeros(R)
    for r_i in range(R):
        # Whole within-lemma label vectors move between equal-size lemma
        # blocks, preserving exact class counts and within-lemma dependence.
        perm = block_permute_labels(labels, lemma_strata, rng)
        scores = np.zeros((len(test_words), len(CLASSES)), dtype=float)
        for k in range(1, 5):
            x_train = train_mats[k]
            x_test = test_mats[k]
            counts = np.vstack(
                [x_train[perm == c].sum(axis=0) for c in range(len(CLASSES))]
            )
            totals = np.array(
                [int(((perm == c) & eligible[k]).sum()) for c in range(len(CLASSES))]
            )
            logp = np.log(
                (counts + ALPHA)
                / (totals[:, None] + ALPHA * max(x_train.shape[1], 1))
            )
            scores += x_test @ logp.T
        pred = scores.argmax(axis=1)
        sims[r_i] = metrics(y_true, pred)["balanced_accuracy"]
    p = float(((sims >= observed).sum() + 1) / (R + 1))
    return p, float(sims.mean()), float(np.quantile(sims, 0.95))


def artifact_key(row: dict) -> str:
    """Best available physical-object proxy for candidate support counts."""
    if (
        (row["src"] == "CIEW" and row["eid"] == "9001")
        or (row["src"] == "CIEP" and row["eid"] == "15910")
        or (row["src"] == "ETP" and row["eid"].startswith("LL"))
    ):
        return "MONUMENT:Liber_Linteus"
    if (
        (row["src"] == "CIEW" and row["eid"] == "7002")
        or (row["src"] == "CIEP" and row["eid"] == "8682")
    ):
        return "MONUMENT:Tabula_Capuana"
    xref = row.get("xref") or {}
    tm = (xref.get("tm") or "").strip()
    if tm:
        return "TM:" + tm
    if row["src"] in ("CIEP", "CIEW-CIE"):
        return "CIE:" + row["eid"]
    return f'{row["src"]}:{row["eid"]}'


def clean_word_stats(
    corpus: dict, mode: str
) -> tuple[dict[str, Counter], dict[str, set[str]], dict[str, set[str]]]:
    forms: dict[str, Counter] = defaultdict(Counter)
    sources: dict[str, set[str]] = defaultdict(set)
    artifacts: dict[str, set[str]] = defaultdict(set)
    for row in corpus_view(corpus):
        # CIEW records marked ocr? have unresolved whole-record reading
        # uncertainty even when individual tokens carry no token-level flag.
        if "ocr?" in row["flags"]:
            continue
        for token in row["toks"]:
            if token["kind"] != "W" or token["flags"] or not clean_surface(token["form"]):
                continue
            word = fold_word(token["form"], mode)
            if len(word) < 3:
                continue
            forms[word][token["form"]] += 1
            sources[word].add(row["src"])
            artifacts[word].add(artifact_key(row))
    return forms, sources, artifacts


def candidate_rows(
    corpus: dict,
    mode: str,
    model: SuffixNB,
    known: set[str],
    matched_calibration: dict[str, dict],
) -> list[dict]:
    forms, sources, artifacts = clean_word_stats(corpus, mode)
    etp_pos_surfaces: set[str] = set()
    with open(os.path.join("data", "ETP_POS.csv"), encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            surface = row.get("Etruscan") or ""
            if clean_surface(surface):
                etp_pos_surfaces.add(fold_word(surface, mode))
    words = sorted(
        word
        for word, counts in forms.items()
        if sum(counts.values()) >= 2 and word not in known
    )
    scores, selected = model.longest_scores(words, min_k=2)
    order = np.argsort(scores, axis=1)
    rows = []
    for i, word in enumerate(words):
        if not selected[i]:
            continue
        best = int(order[i, -1])
        second = int(order[i, -2])
        label = CLASSES[best]
        calibration = matched_calibration[label]
        ci_low, ci_high = calibration["precision_wilson_95ci"]
        suffix_k, suffix_value = selected[i].split(":", 1)
        suffix_support = model.support[int(suffix_k)][suffix_value]
        rows.append(
            {
                "word_folded": word,
                "forms": " ".join(form for form, _ in forms[word].most_common()),
                "freq_clean": sum(forms[word].values()),
                "sources": " ".join(sorted(sources[word])),
                "n_sources": len(sources[word]),
                "n_artifact_clusters": len(artifacts[word]),
                "already_present_in_etp_pos": word in etp_pos_surfaces,
                "pred_case": label,
                "score_margin_nonprobabilistic": float(
                    scores[i, best] - scores[i, second]
                ),
                "validation_precision_matched": calibration["precision"],
                "validation_precision_95ci_low": ci_low,
                "validation_precision_95ci_high": ci_high,
                "validation_predicted_n_matched": calibration["n_predicted"],
                "tir_suffix_train_support_total": sum(suffix_support.values()),
                "tir_suffix_train_support_pred_class": suffix_support[label],
                "tir_suffix_train_class_counts": ";".join(
                    f"{name}:{suffix_support[name]}"
                    for name in CLASSES
                    if suffix_support[name]
                ),
                "shared_raetic_suffix_feature": selected[i],
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            -float(row["score_margin_nonprobabilistic"]),
            -int(row["n_sources"]),
            -int(row["freq_clean"]),
            str(row["word_folded"]),
        ),
    )


def evaluate_mode(mode: str, corpus: dict) -> tuple[dict, SuffixNB, list[tuple[str, str]]]:
    train = load_tir(mode)
    gold, diagnostics = load_etp_gold(mode, corpus)
    words = [word for word, _ in gold]
    y_true = np.array([CLASS_INDEX[label] for _, label in gold])
    model = SuffixNB().fit(train)
    scores = model.scores(words)
    y_pred = scores.argmax(axis=1)
    evidence = model.evidence_mask(words, min_k=1)
    result = metrics_with_abstention(y_true, y_pred, evidence)
    train_class_counts = Counter(label for _, label, _ in train)
    empirical_log_prior = np.log(
        np.array([train_class_counts[name] for name in CLASSES], dtype=float)
        / len(train)
    )
    empirical_prior_pred = (scores + empirical_log_prior).argmax(axis=1)
    empirical_prior_sensitivity = metrics_with_abstention(
        y_true, empirical_prior_pred, evidence
    )
    empirical_prior_sensitivity.update(
        {
            "class_prior": "empirical TIR class frequencies",
            "inferential_status": "descriptive sensitivity only",
        }
    )
    evaluated_words = [word for word, keep in zip(words, evidence) if keep]
    evaluated_y_true = y_true[evidence]
    evaluated_y_pred = y_pred[evidence]
    rng = np.random.default_rng(SEED)
    lo, hi = cluster_bootstrap(
        evaluated_y_true, evaluated_y_pred, evaluated_words, rng
    )
    cluster_p, cluster_mean, cluster_q95 = family_block_label_null(
        evaluated_y_true, evaluated_y_pred, evaluated_words, rng
    )
    train_p, train_mean, train_q95 = training_label_null(
        train,
        evaluated_words,
        evaluated_y_true,
        result["balanced_accuracy"],
        rng,
    )

    # Descriptive robustness check using only TIR entries whose published
    # checklevel is 0 ("nothing to correct").  It is not an additional
    # primary reported test and therefore receives no separate p-value.
    quality_train = load_tir(mode, checklevel_zero_only=True)
    quality_model = SuffixNB().fit(quality_train)
    quality_scores = quality_model.scores(words)
    quality_pred = quality_scores.argmax(axis=1)
    quality_evidence = quality_model.evidence_mask(words, min_k=1)
    quality = metrics_with_abstention(y_true, quality_pred, quality_evidence)
    quality.update(
        {
            "n_raetic_train": len(quality_train),
            "raetic_class_counts": dict(
                Counter(label for _, label, _ in quality_train)
            ),
            "inferential_status": "descriptive sensitivity only",
        }
    )

    # This simpler one-feature model was selected after inspecting the nested
    # suffix model.  It is explicitly post-hoc; its metrics calibrate the
    # exploratory candidate queue but are not independent evidence.
    longest_scores, selected_suffixes = model.longest_scores(words, min_k=2)
    longest_pred = longest_scores.argmax(axis=1)
    longest_evidence = np.array([bool(value) for value in selected_suffixes])
    longest = metrics_with_abstention(y_true, longest_pred, longest_evidence)
    forms, _, _ = clean_word_stats(corpus, mode)
    candidate_like = np.array(
        [
            bool(selected) and sum(forms[word].values()) >= 2
            for word, selected in zip(words, selected_suffixes)
        ],
        dtype=bool,
    )
    longest_candidate_like = metrics_with_abstention(
        y_true, longest_pred, candidate_like
    )
    longest.update(
        {
            "model": "one longest known suffix, k=2..4",
            "selection_status": "post-hoc descriptive",
            "candidate_like_freq_ge_2": {
                **longest_candidate_like,
                "matching_criteria": (
                    "one known TIR suffix with k=2..4 and clean corpus frequency >=2"
                ),
                "selection_status": "post-hoc descriptive calibration",
            },
        }
    )

    result.update(
        {
            "analysis": (
                "primary exploratory validation: uniform-class-prior nested suffix NB on "
                "evidence-covered types only; no preregistered protocol"
            ),
            "class_prior": "uniform across GEN/NOM/PERT",
            "evidence_rule": "at least one training suffix feature with k=1..4",
            "normalization": mode,
            "n_raetic_train": len(train),
            "raetic_class_counts": dict(Counter(label for _, label, _ in train)),
            "n_exact_train_test_surface_overlap": len(
                {word for word, _, _ in train} & set(words)
            ),
            "exact_train_test_surface_overlap": sorted(
                {word for word, _, _ in train} & set(words)
            ),
            "n_etruscan_test": len(gold),
            "etruscan_class_counts": dict(Counter(label for _, label in gold)),
            "n_suffix_features": sum(len(vocab) for vocab in model.vocab.values()),
            "balanced_accuracy_cluster_bootstrap_95ci": [lo, hi],
            "bootstrap_scope": (
                "conditional target-family bootstrap with the fitted TIR "
                "training sample held fixed; not total transfer uncertainty"
            ),
            "training_lemma_block_permutation": {
                "R": R,
                "p_raw": train_p,
                "p_bonferroni_4": min(1.0, train_p * 4),
                "null_mean": train_mean,
                "null_q95": train_q95,
                "n_lemma_blocks": len(
                    {lemma or word for word, _, lemma in train}
                ),
                "method": (
                    "permute whole label vectors among equal-size lemma blocks; "
                    "global class marginals preserved"
                ),
            },
            "target_family_block_permutation": {
                "R": R,
                "p_raw": cluster_p,
                "p_bonferroni_4": min(1.0, cluster_p * 4),
                "null_mean": cluster_mean,
                "null_q95": cluster_q95,
                "family_rule": "strip longest of " + ",".join(CASE_SUFFIXES),
                "n_families": len(
                    {family_key(word) for word in evaluated_words}
                ),
                "method": (
                    "permute whole label vectors among equal-size target-family "
                    "blocks; global class marginals preserved"
                ),
            },
            "multiple_testing": (
                "Bonferroni correction across four reported tests: two "
                "normalizations x two block-permutation nulls"
            ),
            "tir_checklevel_0_sensitivity": quality,
            "empirical_class_prior_sensitivity": empirical_prior_sensitivity,
            "posthoc_longest_suffix_k2_4": longest,
            **diagnostics,
        }
    )
    return result, model, gold


def main() -> None:
    os.makedirs("logs", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    corpus_path = os.path.join("data", "etr_corpus.pkl")
    declared_path = os.path.join("data", "etr_corpus.sha256")
    corpus_sha = sha256_file(corpus_path)
    declared_sha = open(declared_path, encoding="utf-8").read().split()[0]
    assert corpus_sha == declared_sha, "etr_corpus.pkl does not match declared SHA-256"
    corpus = pickle.load(open(corpus_path, "rb"))
    assert corpus["meta"].get("freeze_version") == "0.7"
    input_paths = {
        "corpus": corpus_path,
        "tir_words": os.path.join("data", "external", "tir", "tir_words.csv"),
        "etp_pos": os.path.join("data", "ETP_POS.csv"),
        "script": os.path.abspath(__file__),
    }
    results = {
        "meta": {
            "seed": SEED,
            "permutations": R,
            "alpha": ALPHA,
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "input_sha256": {
                name: sha256_file(path) for name, path in input_paths.items()
            },
            "declared_corpus_sha256": declared_sha,
        }
    }
    primary_model = None
    primary_gold = None
    for mode in ("strict", "minimal"):
        result, model, gold = evaluate_mode(mode, corpus)
        results[mode] = result
        log(f"=== Raetic -> Etruscan case transfer ({mode}) ===")
        log(
            f"TIR train={result['n_raetic_train']} {result['raetic_class_counts']}; "
            f"ETP gold={result['n_etruscan_test']} {result['etruscan_class_counts']}"
        )
        log(
            f"evidence coverage={result['n_evaluated']}/{result['n_total']} "
            f"({result['coverage']:.3f}); abstained={result['n_abstained']}"
        )
        log(
            f"covered accuracy={result['accuracy']:.3f}; "
            f"balanced={result['balanced_accuracy']:.3f} "
            f"(cluster bootstrap 95% {result['balanced_accuracy_cluster_bootstrap_95ci'][0]:.3f}"
            f"..{result['balanced_accuracy_cluster_bootstrap_95ci'][1]:.3f}); "
            f"macro-F1={result['macro_f1']:.3f}"
        )
        log(f"confusion rows={CLASSES}, cols={CLASSES}: {result['confusion_rows_true_cols_pred']}")
        for name in CLASSES:
            values = result["per_class"][name]
            log(
                f"  {name}: n={values['n']} precision={values['precision']:.3f} "
                f"recall={values['recall']:.3f} F1={values['f1']:.3f}"
            )
        perm = result["training_lemma_block_permutation"]
        clustered = result["target_family_block_permutation"]
        log(
            f"train lemma-block null R={R}: mean={perm['null_mean']:.3f}, "
            f"q95={perm['null_q95']:.3f}, p_raw={perm['p_raw']:.5f}, "
            f"p_Bonf4={perm['p_bonferroni_4']:.5f}"
        )
        log(
            f"target family-block null ({clustered['n_families']} families): "
            f"mean={clustered['null_mean']:.3f}, q95={clustered['null_q95']:.3f}, "
            f"p_raw={clustered['p_raw']:.5f}, "
            f"p_Bonf4={clustered['p_bonferroni_4']:.5f}"
        )
        quality = result["tir_checklevel_0_sensitivity"]
        log(
            f"checklevel=0 sensitivity: train={quality['n_raetic_train']}, "
            f"coverage={quality['n_evaluated']}/{quality['n_total']} "
            f"({quality['coverage']:.3f}), balanced={quality['balanced_accuracy']:.3f}"
        )
        longest = result["posthoc_longest_suffix_k2_4"]
        matched = longest["candidate_like_freq_ge_2"]
        log(
            f"POST-HOC longest k=2..4: coverage={longest['n_evaluated']}/"
            f"{longest['n_total']} ({longest['coverage']:.3f}), "
            f"accuracy={longest['accuracy']:.3f}, balanced={longest['balanced_accuracy']:.3f}"
        )
        log(
            f"POST-HOC candidate-like freq>=2: n={matched['n_evaluated']}, "
            f"accuracy={matched['accuracy']:.3f}, balanced={matched['balanced_accuracy']:.3f}"
        )
        empirical = result["empirical_class_prior_sensitivity"]
        log(
            f"empirical-prior sensitivity: accuracy={empirical['accuracy']:.3f}, "
            f"balanced={empirical['balanced_accuracy']:.3f}"
        )
        log()
        if mode == "minimal":
            primary_model, primary_gold = model, gold

    assert primary_model is not None and primary_gold is not None
    primary = results["minimal"]
    matched_calibration = primary["posthoc_longest_suffix_k2_4"][
        "candidate_like_freq_ge_2"
    ]["per_class"]
    candidates = candidate_rows(
        corpus,
        "minimal",
        primary_model,
        {word for word, _ in primary_gold},
        matched_calibration,
    )
    fields = [
        "word_folded",
        "forms",
        "freq_clean",
        "sources",
        "n_sources",
        "n_artifact_clusters",
        "already_present_in_etp_pos",
        "pred_case",
        "score_margin_nonprobabilistic",
        "validation_precision_matched",
        "validation_precision_95ci_low",
        "validation_precision_95ci_high",
        "validation_predicted_n_matched",
        "tir_suffix_train_support_total",
        "tir_suffix_train_support_pred_class",
        "tir_suffix_train_class_counts",
        "shared_raetic_suffix_feature",
    ]
    with open(OUT_CSV, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in candidates:
            row = dict(row)
            for field in (
                "score_margin_nonprobabilistic",
                "validation_precision_matched",
                "validation_precision_95ci_low",
                "validation_precision_95ci_high",
            ):
                row[field] = (
                    "" if row[field] is None else f"{float(row[field]):.6f}"
                )
            writer.writerow(row)
    results["candidate_output"] = {
        "path": OUT_CSV,
        "n": len(candidates),
        "n_already_present_in_etp_pos": sum(
            bool(row["already_present_in_etp_pos"]) for row in candidates
        ),
        "predicted_class_counts": dict(Counter(row["pred_case"] for row in candidates)),
        "n_tir_suffix_support_ge_2": sum(
            int(row["tir_suffix_train_support_total"]) >= 2 for row in candidates
        ),
        "n_artifact_clusters_ge_2": sum(
            int(row["n_artifact_clusters"]) >= 2 for row in candidates
        ),
        "n_both_support_ge_2": sum(
            int(row["tir_suffix_train_support_total"]) >= 2
            and int(row["n_artifact_clusters"]) >= 2
            for row in candidates
        ),
        "model": "post-hoc one longest known TIR suffix, k=2..4",
        "eligibility": (
            "clean corpus frequency >=2 and not in clean single-case ETP_POS gold; "
            "already_present_in_etp_pos flags other ETP_POS occurrences; "
            "support/artifact columns expose the main uncertainty filters"
        ),
        "score_definition": (
            "difference between top two smoothed suffix log-likelihoods; not a "
            "probability, calibrated confidence, or individual test statistic"
        ),
        "calibration": (
            "post-hoc ETP_POS subset matching suffix evidence and clean frequency >=2"
        ),
        "warning": (
            "Exploratory ranking queue, not deciphered meanings. No candidate has an "
            "individual hypothesis test or FDR control; post-hoc class precision may "
            "not transport to unlabeled types."
        ),
    }
    with open(OUT_JSON, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    log(f"candidates={len(candidates)} -> {OUT_CSV}")
    log(f"validation -> {OUT_JSON}; log -> {OUT_LOG}")
    with open(OUT_LOG, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
