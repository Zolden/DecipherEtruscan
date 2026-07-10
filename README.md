# DecipherEtruscan: extending the understood share of Etruscan, statistically

A sibling project of
[DecipherLinearA](https://github.com/Zolden/DecipherLinearA)
(DOI 10.5281/zenodo.21262275), transferring its reproducible statistical
method — frozen, provenance-checked corpus (manual validation pending), permutation null models, calibrated
morphology, slot-template engines, honest negatives — to Etruscan.

Etruscan differs from Linear A in the decisive way: the script reads
reliably and a few hundred words are already understood, so for the first
time our "meaning-inference" methods get a **validation set** (~33% of
inscriptions carry translations in the compiled dataset). The goal is not
full decipherment (the small Tyrsenian family has no close high-resource
language and 81% of word types are hapaxes) but a measurable extension of
understanding: a complete formal
morphology, ranked semantic-class hypotheses evaluated once on untouched
artifact-level test sets, and a structural (slot-level) reading of the long ritual texts,
Liber Linteus above all.

Current frozen view (v0.6, audited 2026-07-10): 6,795 Etruscan records,
12,450 word tokens, 7,384 types (81% hapaxes), and 2,238 records with source
glosses/translations. Operator and morphology results, including retractions
and cluster-aware robustness checks, are documented in `etruscan_report.md`
and `METHOD_AUDIT_SOL_20260710.md`. Data: Larth-Etruscan-NLP compilation
(Vico & Spanakis 2023) + ETP/CIEP derivatives — see `data/` and credits in
`CLAUDE.md`. Third-party provenance, licence conflicts, and redistribution
limits are catalogued in [`DATA_SOURCES.md`](DATA_SOURCES.md).

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21279781.svg)](https://doi.org/10.5281/zenodo.21279781)

Cite: Zolden & Claude (2026). DecipherEtruscan: statistical extension of
Etruscan understanding (v0.6.1). Zenodo. doi:10.5281/zenodo.21279781

Status: frozen, provenance-checked corpus
(`data/etr_corpus.pkl`: 7,338 records total, 6,795 in the clean Etruscan
text view; see `etruscan_report.md` §§0 and 8). Work proceeds in staged,
fully reproducible
steps (seed=42, PYTHONHASHSEED=0; the frozen analytical pipeline gives an
empty `git diff` after a full rerun; network refreshers separately record
their retrieval time and exact upstream revisions).
