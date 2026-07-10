# U.S. Epigraphy Project: Etruscan/Raetic EpiDoc subset

Pinned subset of the [U.S. Epigraphy Project](https://usepigraphy.brown.edu/)
EpiDoc repository maintained by Brown University Library.  It contains every
upstream XML record whose `textLang/@mainLang` was `ett` (Etruscan), plus the
single `xrr` (Raetic) record, at upstream commit
`d34663bb2b772e7291851d08669a6565ebcc01c0` (2026-06-10).

Upstream repository:
<https://github.com/Brown-University-Library/usep-data>

Retrieved: 2026-07-10.  The subset has 35 Etruscan records: 26 with EpiDoc
editions and 9 metadata-only records.  The one Raetic record,
`RI.Prov.RISD.MA.Raet.32.245`, is metadata-only and has no transcription; it
is therefore a provenance/object cross-check, not an addition to the TIR text
corpus.  `resources/` contains the upstream bibliography and controlled
taxonomies referenced by XInclude.  Run `python extract_manifest.py` to
rebuild `manifest.csv` from the XML.

## Why it is useful

USEP supplies object-level museum provenance, date ranges, support/material,
bibliography, image links, and (for 26 Etruscan records) independently grouped
texts and some English translations.  Several mirror labels that occur as
isolated CIEP entries are grouped here on the same object.  This is useful for
object-aware co-occurrence, iconographic name alignment, and independent
checking of funerary/name formulae.

Do not merge `manifest.csv` blindly into the frozen corpus.  Some editions
expand abbreviations, one is encoded directly in Old Italic Unicode, readings
and translations have mixed levels of specialist verification, and the
metadata-only set includes uncertain or very broad dates.  Selection here is
based on the upstream `textLang/@mainLang` tag, not a fresh linguistic
adjudication: the set includes iconographic Greek-name labels and apparently
Latin or expanded forms.  The XML, revision history, bibliography and linked
museum page should be checked before any claim.  The upstream site's published
count (22 Etruscan records) is stale relative to the pinned Git tree, which
contains 35 records tagged `ett`.

## License and attribution

The USEP website states that the work is licensed under
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/):
<https://usepigraphy.brown.edu/projects/usep/about/>.  Attribute the U.S.
Epigraphy Project/Brown University and the individual encoders and cited
editions recorded in each XML file; retain the same license for derivatives.

Checksums are in `SHA256SUMS`.
