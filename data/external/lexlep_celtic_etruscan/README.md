# Lexicon Leponticum: Etruscan inscriptions with Celtic material

Small, pinned extract of the eleven records returned by Lexicon Leponticum's
Semantic MediaWiki query for `[[Category:Inscription]][[corpus::Etruscan]]`.
This is **not** the general Etruscan corpus: LexLep includes these records
because they contain, or have been argued to contain, Celtic onomastic or
linguistic material.  That makes the extract useful for contact-sensitive
onomastics and as an independently edited check on readings, dates, genre and
provenance.

Source: <https://lexlep.univie.ac.at/wiki/Inscriptions>

API: <https://lexlep.univie.ac.at/api.php>

Retrieved: 2026-07-10.  `ask_records.json` is the structured Semantic
MediaWiki result; `pages_wikitext.json` preserves the eleven page revisions,
including commentary and bibliography; `siteinfo_rights.json` records the
site's machine-readable rights statement.  Run `python extract_records.py` to
rebuild the convenience table `records.csv` from the pinned JSON.

## Scope and cautions

The records are AS 3.1, AS 3.2, Cl 3.2, Cr 3.22, Li 1.1, Li 1.2, Pa 0.3,
Pa 1.2, SH·1, Vs 1.165 and Vs 1.87.  `text_plain` is LexLep's unsegmented
search form, not a diplomatic edition.  Read the page wikitext before using a
damaged or disputed form.  In particular, SH·1 is explicitly only a possible
language-encoding inscription, and the page itself gives substantial reasons
to prefer a non-linguistic/numerical interpretation.  LexLep `meaning` values
are editorial interpretations, not gold-standard translations.

An exact-ID/text check against this repository's pre-existing Larth/ETP/CIEP
files on 2026-07-10 found clear local counterparts for Cl 3.2, Li 1.1 and Li
1.2.  The remaining records require manual concordance before being treated as
new inscriptions; absence of an exact string is not proof of corpus absence.

## License and attribution

Lexicon Leponticum's notices are inconsistent. The site footer and pinned API
`siteinfo_rights.json` state
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/), while
the dedicated [terms page](https://lexlep.univie.ac.at/wiki/Project:Terms_of_use)
offers CC BY-SA 3.0 and GFDL. This snapshot conservatively applies the more
restrictive CC BY-NC-SA 4.0 unless LexLep clarifies otherwise. Attribute
Lexicon Leponticum and the individual page editors/source publications;
retain share-alike terms for derivatives. Project overview and editorial conventions:
<https://lexlep.univie.ac.at/wiki/How_to_use_LexLep>.

Checksums are in `SHA256SUMS`.
