# DecipherEtruscan: extending the understood share of Etruscan, statistically

A sibling project of
[DecipherLinearA](https://github.com/Zolden/DecipherLinearA)
(DOI 10.5281/zenodo.21262275), transferring its reproducible statistical
method — frozen validated corpus, permutation null models, calibrated
morphology, slot-template engines, honest negatives — to Etruscan.

Etruscan differs from Linear A in the decisive way: the script reads
reliably and a few hundred words are already understood, so for the first
time our "meaning-inference" methods get a **validation set** (~42% of
inscriptions carry translations in the compiled dataset). The goal is not
full decipherment (the language is an isolate and 78% of word types are
hapaxes) but a measurable extension of understanding: a complete formal
morphology, ranked probabilistic meanings for unknown words with test-set
accuracy, and a structural (slot-level) reading of the long ritual texts,
Liber Linteus above all.

Pilot measurements (see `CLAUDE.md`): 5,842 inscriptions with text, 96%
covered by the top-10 templates, operator words profile instantly
(mi "I" 89% initial; lupu "died" 80% final), genitive paradigms are
massive (-s: 876 types). Data: Larth-Etruscan-NLP compilation
(Vico & Niculae 2023) + ETP/CIEP derivatives — see `data/` and credits in
`CLAUDE.md`.

Status: stage 0 done — frozen, validated corpus
(`data/etr_corpus.pkl`: 6,361 records, 6,238 in the Etruscan text view;
see `etruscan_report.md` §0). Work proceeds in staged, fully reproducible
steps (seed=42, PYTHONHASHSEED=0; empty `git diff` after full rerun).
