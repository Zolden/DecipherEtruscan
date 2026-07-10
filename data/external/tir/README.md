# Thesaurus Inscriptionum Raeticarum (TIR) — reproducible tables

This directory is a machine-readable snapshot of the **Thesaurus
Inscriptionum Raeticarum**, the University of Vienna's scholarly digital
edition of the Raetic corpus.  Raetic is a Tyrsenian language and is the most
important external comparative control for Etruscan morphology.

Upstream: https://tir.univie.ac.at/

Terms: https://tir.univie.ac.at/wiki/Project:Terms_of_Use

Required attribution: *Thesaurus Inscriptionum Raeticarum*, ed. Stefan
Schumacher, Corinna Salomon, Sindy Kluge, Gudrun Bajc & Martin Braun,
2013–.  Each row retains its upstream URL, exact revision id and revision
timestamp.

## Licence

TIR states that its textual content is available under **CC BY-SA 3.0** and
the **GNU Free Documentation License**.  These derived tables are therefore
distributed under CC BY-SA 3.0 with the attribution above.  No images were
downloaded; image rights vary by holding institution and are outside this
snapshot.

## Files and scope (retrieved 2026-07-10)

- `tir_inscriptions.csv`: 389 inscription records.  It joins 386 records to
  their object page, adding object type/material, date, site and collection.
  Only **112** rows are explicitly classified by TIR as `Raetic`; 262 are
  `unknown`, 13 `none`, and 2 Latin.  Do not train a Raetic language model on
  all 389 rows without respecting `language`.
- `tir_objects.csv`: 293 real object records (the category's synthetic
  `unknown` page is intentionally omitted).
- `tir_words.csv`: 154 real lexical records (likewise excluding the synthetic
  `unknown` page), including lemma, word type, case, gender, morphemic
  analysis and cautious meanings.
- `tir_morphemes.csv`: 11 explicitly analysed morphemes.
- `metadata.json`: retrieval time, row counts, source/licence and SHA-256
  hashes.
- `SHA256SUMS`: checksums for the generated snapshot.

`reading_raw` preserves TIR's internal `link-target!display-form` markup after
HTML-entity decoding.  `reading` removes link/control targets, drops the
non-displayed inline alternative used by a few damaged readings, and converts
TIR's `space` archigrapheme to a non-breaking space.  Ordinary template
whitespace is removed because it is not an inscription word boundary.  The
fetcher verifies every result against TIR's rendered plain-text category table;
this snapshot matches **389/389** readings exactly.  Editorial uncertainty
signs and the `$` symbol archigrapheme are preserved.
`sortdate` is TIR's sortable numeric date (BC values are negative); `date`
and `date_derivation` retain the human-readable dating and its stated basis.
TIR's `checklevel` is an editorial to-do priority, not an uncertainty score:
`0` means “nothing to correct”, while higher values mean more work remains
(see the upstream `Property:checklevel` page).

## Reproduction

From the repository root:

```powershell
$env:PYTHONIOENCODING='utf-8'
.\.venv\Scripts\python.exe tools\etr_fetch_tir.py
```

The fetcher uses only Python's standard library and the official public
MediaWiki API.  It records the exact revision used for every page and emits
LF-normalised CSV.  A later upstream revision will legitimately change the
checksums; preserve this snapshot for analyses that must remain byte-stable.

## Intended analytical use

The safe comparison set is `language == "Raetic"`, with sensitivity analyses
that exclude damaged/uncertain readings and deduplicate multi-inscription
objects.  The word and morpheme tables provide external labels for testing
Etruscan suffix hypotheses (not for importing Raetic analyses into Etruscan as
facts).  Particularly useful held-out controls are Raetic patronymic `-nu`,
feminine `-na`, genitive `-s(i)`, pertinentive `-si/-le`, and past `-ke`.
