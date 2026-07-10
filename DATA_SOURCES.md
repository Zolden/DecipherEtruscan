# Data sources, licences, and provenance

The repository-level MIT licence applies to original code and prose only. It
does **not** relicense ancient-text compilations, extracted databases, source
snapshots, or scholarly PDFs. Each third-party component retains the terms
below and the more specific terms in its local README. The frozen
`data/etr_corpus.pkl` is a mixed derived research artefact and is not offered
under MIT as a standalone corpus.

## Core inputs

| Files | Source and status |
|---|---|
| `data/etruscan_larth.csv`, `ETP_fix.csv`, `ETP_POS.csv`, `CIEP_pymupdf.csv` | Derived from Larth-Etruscan-NLP / Etruscan Texts Project / CIE material; cite Vico & Spanakis (2023), [ACL Anthology](https://aclanthology.org/2023.alp-1.5/). The upstream code/data repository did not state a reusable data licence when checked. Included for research verification with attribution; redistribution permission should be confirmed before repackaging these files as a dataset. |
| `data/external/burman/` | Kevin Burman's digital concordance, version 1.0.3; CC0. The bundled `legalcode.txt`, readme, edit log, and exact version identify the release. |
| `data/external/fowler_wolfe/ciew_*.csv` | Project extraction of public-domain ancient inscription strings from Fowler & Wolfe (1965). The modern book/OCR is likely copyrighted until about 2061 and is deliberately excluded; see the local README. Treat the derived tables as research material pending a definitive rights review. |
| `data/supplements/*.csv` | Small attributed extracts from the named Wikipedia pages, whose text is CC BY-SA/GFDL. Each row carries its URL and retrieval note. Any later supplement must provide non-empty `provenance`. |
| `data/concepts/concept_lexicon.csv` | Project-generated hypothesis lexicon and table structure. The forms were model-assisted and lack row-level scholarly provenance; use only as an exploratory input, not as a reference lexicon. |
| `data/external/texrel/` | Legacy list of queried non-Etruscan TM identifiers. The original response bodies were not preserved, so it is not evidence of a negative API result; see the local README. Trismegistos data-service terms apply. |

The upstream absence of an explicit licence is not converted into permission
by this repository. Users who redistribute a combined corpus must perform
their own rights review and preserve source attribution.

## Added open external snapshots

| Directory | Licence and required attribution |
|---|---|
| `data/external/tir/` | Derived textual tables from *Thesaurus Inscriptionum Raeticarum*, ed. Stefan Schumacher, Corinna Salomon, Sindy Kluge, Gudrun Bajc & Martin Braun, 2013–; CC BY-SA 3.0 and GFDL. No images are included. See its README and [TIR terms](https://tir.univie.ac.at/wiki/Project:Terms_of_Use). |
| `data/external/usep_tyrsenian/` | Pinned EpiDoc subset of the U.S. Epigraphy Project / Brown University Library; CC BY-NC-SA 4.0 as stated by the project. Attribute USEP, individual encoders, and cited editions. |
| `data/external/lexlep_celtic_etruscan/` | Pinned API extract of *Lexicon Leponticum*. Its footer/API says CC BY-NC-SA 4.0, while the dedicated terms page says CC BY-SA 3.0/GFDL; this snapshot conservatively follows the more restrictive CC BY-NC-SA 4.0 pending clarification. Attribute LexLep and page editors/sources. |
| `data/external/etruscan_reference/` | Belfiore (2020), CC BY-NC-SA; unchanged scholarly PDF. |
| `data/external/raetic_reference/` | Salomon (2020) overview, CC BY-NC-SA; Salomon (2020) personal-name paper, CC BY 3.0. |
| `data/external/computational_decipherment_reference/` | Braović et al. (2024), *Computational Linguistics* 50(2), article-specific **CC BY-NC-ND 4.0** as printed on the PDF. The included PDF is unchanged; commercial or modified redistribution is not permitted by that licence. |

Every directory above contains a README with source URL/DOI, retrieval date,
scope cautions, licence details, and `SHA256SUMS`. Share-alike and
non-commercial restrictions continue to apply to the relevant components and
their derivatives; inclusion here does not place them under MIT.

## Results and takedown

Project-authored code, statistical summaries, and original commentary are MIT
unless marked otherwise. Candidate lists are analytical outputs, not
translations or expert editions, and remain subject to the provenance limits
of their inputs.

If a rights holder identifies material that should not be redistributed,
please open a repository issue with the file path and evidence of ownership.
The material will be isolated or removed while the claim is reviewed.
