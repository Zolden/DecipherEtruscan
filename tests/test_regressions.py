from __future__ import annotations

import csv
import hashlib
import json
import pickle
import unittest
from pathlib import Path

from tools import etr_freeze
from tools import etr_method_audit_sol as method_audit
from tools import etr_raetic_transfer as raetic_transfer


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class CorpusRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with (ROOT / "data" / "etr_corpus.pkl").open("rb") as handle:
            cls.corpus = pickle.load(handle)

    def test_lemnos_cie_is_not_etruscan_training_data(self) -> None:
        self.assertEqual(etr_freeze.LANG_BY_CIE["15999"][0], "lemn")
        rows = [
            row
            for row in self.corpus["records"]
            if row["src"] == "CIEP" and row["eid"] == "15999"
        ]
        self.assertEqual(len(rows), 2)
        self.assertEqual({row["lang"] for row in rows}, {"lemn"})

    def test_frozen_corpus_matches_declared_hash(self) -> None:
        declared = (ROOT / "data" / "etr_corpus.sha256").read_text(
            encoding="utf-8"
        ).split()[0]
        self.assertEqual(sha256(ROOT / "data" / "etr_corpus.pkl"), declared)

    def test_supplements_have_nonempty_provenance(self) -> None:
        for path in sorted((ROOT / "data" / "supplements").glob("*.csv")):
            with path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertTrue(rows, path.name)
            for row in rows:
                self.assertTrue((row.get("text") or "").strip(), path.name)
                self.assertTrue((row.get("provenance") or "").strip(), path.name)


class AnalysisRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with (ROOT / "data" / "etr_corpus.pkl").open("rb") as handle:
            cls.corpus = pickle.load(handle)

    def test_known_monument_aliases_share_artifact_key(self) -> None:
        liber_rows = [
            {"src": "ETP", "eid": "LL 2", "xref": {}},
            {"src": "CIEP", "eid": "15910", "xref": {}},
            {"src": "CIEW", "eid": "9001", "xref": {}},
        ]
        capua_rows = [
            {"src": "CIEP", "eid": "8682", "xref": {}},
            {"src": "CIEW", "eid": "7002", "xref": {}},
        ]
        self.assertEqual(
            {raetic_transfer.artifact_key(row) for row in liber_rows},
            {"MONUMENT:Liber_Linteus"},
        )
        self.assertEqual(
            {method_audit.artifact_id(row) for row in capua_rows},
            {"MONUMENT:Tabula_Capuana"},
        )

    def test_tied_etp_gold_is_excluded(self) -> None:
        _, etp = method_audit.semantic_labels([])
        self.assertNotIn("larthia", etp)
        self.assertNotIn("larthial", etp)

    def test_unresolved_ocr_only_forms_are_not_clean_candidates(self) -> None:
        forms, _, _ = raetic_transfer.clean_word_stats(self.corpus, "minimal")
        for word in ("zusle", "nunoeri", "ilukve"):
            self.assertNotIn(word, forms)
        # Resolved ocr-fixed records are deliberately retained.
        self.assertIn("θezine", forms)
        self.assertIn("latiθe", forms)

    def test_transfer_artifact_is_self_describing(self) -> None:
        result = json.loads(
            (ROOT / "results" / "raetic_transfer_validation.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(result["candidate_output"]["n"], 265)
        self.assertEqual(result["candidate_output"]["n_both_support_ge_2"], 91)
        self.assertEqual(
            result["meta"]["input_sha256"]["corpus"],
            sha256(ROOT / "data" / "etr_corpus.pkl"),
        )

    def test_morphology_signal_survives_clean_reading_sensitivity(self) -> None:
        result = json.loads(
            (ROOT / "results" / "method_audit_sol_20260710.json").read_text(
                encoding="utf-8"
            )
        )
        clean = result["morphology"]["clean_reading_longest_match_sensitivity"]
        self.assertEqual(clean["tested_pairs"], 91)
        self.assertEqual(clean["reported_ten_bonferroni_lt_05"], 5)
        s_al = next(
            row for row in clean["reported_ten"] if row["pair"] == ["s", "al"]
        )
        self.assertEqual(s_al["observed"], 4)
        self.assertAlmostEqual(s_al["expected"], 15.7696, places=4)
        self.assertLess(s_al["p_bonf_longest_all"], 0.002)


class ExternalSnapshotTests(unittest.TestCase):
    def test_all_external_checksums(self) -> None:
        for manifest in sorted((ROOT / "data" / "external").rglob("SHA256SUMS")):
            for line in manifest.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                expected, name = line.split("  ", 1)
                self.assertEqual(sha256(manifest.parent / name), expected, name)

    def test_legacy_texrel_file_is_valid_but_has_no_claimed_responses(self) -> None:
        path = ROOT / "data" / "external" / "texrel" / "texrel_nonetr.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(rows), 242)
        self.assertEqual(len({row["tm"] for row in rows}), 242)
        self.assertTrue(all(row["response"] is None for row in rows))
        self.assertEqual(
            {row["status"] for row in rows}, {"legacy_response_not_preserved"}
        )


if __name__ == "__main__":
    unittest.main()
