"""
Unit Tests for Phase 6: Post-Processing, Test Inference & Submission Verification
==================================================================================
Tests end-to-end inference output formatting, delimiter compliance, singleton
representation, candidate superset invariants, zero-shot France processing,
submission validation pass, and packaging archive verification.
"""

from pathlib import Path
import subprocess
import sys
import unittest
import zipfile

import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.matching_model import run_prediction_pipeline
from utils.package_submission import package_submission


class TestPhase6SubmissionVerification(unittest.TestCase):
    """Test suite for Phase 6 inference, validation, and packaging."""

    @classmethod
    def setUpClass(cls):
        """Execute test prediction pipeline to generate fresh output artefacts."""
        cls.output_dir = PROJECT_ROOT / "output"
        cls.test_dir = PROJECT_ROOT / "dataset" / "test"
        cls.matching_file, cls.candidate_file = run_prediction_pipeline()

    def test_output_files_exist_and_non_empty(self):
        """Verify matching_results.tsv and candidate_pairs.tsv exist and have data."""
        self.assertTrue(self.matching_file.exists(), f"Missing {self.matching_file}")
        self.assertTrue(self.candidate_file.exists(), f"Missing {self.candidate_file}")
        self.assertGreater(self.matching_file.stat().st_size, 0)
        self.assertGreater(self.candidate_file.stat().st_size, 0)

    def test_strict_tab_delimiters_and_no_quotes(self):
        """Verify tab separation and strict absence of quotation marks in both TSV files."""
        for path in [self.matching_file, self.candidate_file]:
            with open(path, "r", encoding="utf-8") as f:
                for line_idx, line in enumerate(f, start=1):
                    # Check delimiter
                    self.assertIn("\t", line, f"Missing tab delimiter at {path.name}:{line_idx}")
                    parts = line.split("\t")
                    self.assertEqual(len(parts), 2, f"Line {line_idx} has {len(parts)} columns, expected 2")
                    # Check no quotes
                    self.assertNotIn('"', line, f"Illegal quote detected at {path.name}:{line_idx}")
                    self.assertNotIn("'", line, f"Illegal quote detected at {path.name}:{line_idx}")

    def test_exact_headers(self):
        """Verify exact expected column headers for both files."""
        with open(self.matching_file, "r", encoding="utf-8") as f:
            header_m = f.readline().rstrip("\r\n").split("\t")
            self.assertEqual(header_m, ["source1_entity_id", "matched_entity_ids"])

        with open(self.candidate_file, "r", encoding="utf-8") as f:
            header_c = f.readline().rstrip("\r\n").split("\t")
            self.assertEqual(header_c, ["source1_entity_id", "candidate_entity_ids"])

    def test_all_test_s1_entities_present_and_unique(self):
        """Verify 100% of test Source 1 entities are present and have zero duplicate rows."""
        s1_path = self.test_dir / "test_source1.tsv"
        df_s1 = pd.read_csv(s1_path, sep="\t", dtype=str, keep_default_na=False)
        expected_s1_ids = set(df_s1["entity_id"])

        df_m = pd.read_csv(self.matching_file, sep="\t", dtype=str, keep_default_na=False)
        df_c = pd.read_csv(self.candidate_file, sep="\t", dtype=str, keep_default_na=False)

        # Equal row count
        self.assertEqual(len(df_m), len(df_s1))
        self.assertEqual(len(df_c), len(df_s1))

        # Exact ID matching
        self.assertEqual(set(df_m["source1_entity_id"]), expected_s1_ids)
        self.assertEqual(set(df_c["source1_entity_id"]), expected_s1_ids)

        # Zero duplicates
        self.assertEqual(df_m["source1_entity_id"].nunique(), len(df_m))
        self.assertEqual(df_c["source1_entity_id"].nunique(), len(df_c))

    def test_candidate_superset_invariant(self):
        """Ensure every predicted match is present in the corresponding candidate list."""
        df_m = pd.read_csv(self.matching_file, sep="\t", dtype=str, keep_default_na=False)
        df_c = pd.read_csv(self.candidate_file, sep="\t", dtype=str, keep_default_na=False)

        cand_map = dict(zip(df_c["source1_entity_id"], df_c["candidate_entity_ids"]))

        for _, row in df_m.iterrows():
            sid = row["source1_entity_id"]
            matched_str = row["matched_entity_ids"]
            if matched_str:
                matched_set = set(matched_str.split(","))
                cand_str = cand_map.get(sid, "")
                cand_set = set(cand_str.split(",")) if cand_str else set()
                self.assertTrue(
                    matched_set.issubset(cand_set),
                    f"Candidate superset violation for {sid}: {matched_set - cand_set} not in candidates",
                )

    def test_no_self_matches_or_invalid_prefixes(self):
        """Assert no S1 self-matches and all IDs have valid S2- or S3- prefixes."""
        for path, col in [(self.matching_file, "matched_entity_ids"), (self.candidate_file, "candidate_entity_ids")]:
            df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
            for _, row in df.iterrows():
                val = row[col]
                if val:
                    for target_id in val.split(","):
                        self.assertFalse(
                            target_id.startswith("S1-"),
                            f"Illegal self-match: {target_id} in {path.name}",
                        )
                        self.assertTrue(
                            target_id.startswith(("S2-", "S3-")),
                            f"Invalid entity ID prefix: {target_id} in {path.name}",
                        )

    def test_singleton_formatting(self):
        """Verify singletons emit empty string immediately preceded by tab (S1-xxxxx\\t\\n)."""
        with open(self.matching_file, "r", encoding="utf-8") as f:
            lines = [line.rstrip("\r\n") for line in f]

        # S1-10005 and S1-10010 are singletons in the test sample
        singleton_ids = {"S1-10005", "S1-10010"}
        for line in lines[1:]:
            parts = line.split("\t")
            if parts[0] in singleton_ids:
                self.assertEqual(parts[1], "", f"Expected empty string for singleton {parts[0]}, got '{parts[1]}'")

    def test_zero_shot_france_processing(self):
        """Verify France records (S1-10001 to S1-10005) are processed without exceptions."""
        df_m = pd.read_csv(self.matching_file, sep="\t", dtype=str, keep_default_na=False)
        french_s1_ids = {"S1-10001", "S1-10002", "S1-10003", "S1-10004", "S1-10005"}
        found_french = set(df_m["source1_entity_id"]) & french_s1_ids
        self.assertEqual(found_french, french_s1_ids, "Not all French entities were processed in matching results")

    def test_official_validator_pass(self):
        """Run utils/validate_submission.py and assert exit code 0 (PASS)."""
        validator = PROJECT_ROOT / "utils" / "validate_submission.py"
        cmd = [
            sys.executable,
            str(validator),
            "--matching", str(self.matching_file),
            "--candidate", str(self.candidate_file),
            "--test-dir", str(self.test_dir),
            "--check-ids",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, f"Validator failed with output:\n{res.stdout}\n{res.stderr}")
        self.assertIn("PASS", res.stdout)

    def test_submission_packaging_and_manifest(self):
        """Verify package_submission utility produces valid zip with exact contest structure."""
        zip_path = PROJECT_ROOT / "test_package_submission.zip"
        if zip_path.exists():
            zip_path.unlink()

        created_zip = package_submission(
            team_name="test_team",
            output_zip=zip_path,
            skip_validation=False,
            test_dir=self.test_dir,
        )
        self.assertTrue(created_zip.exists())
        self.assertGreater(created_zip.stat().st_size, 0)

        # Audit internal zip manifest
        expected_entries = {
            "output/matching_results.tsv",
            "output/candidate_pairs.tsv",
            "Documentation_template.md",
            "code/business_entity_resolution/README.md",
            "code/business_entity_resolution/requirements.txt",
            "code/business_entity_resolution/src/__init__.py",
            "code/business_entity_resolution/src/blocking.py",
            "code/business_entity_resolution/src/data_cleaning.py",
            "code/business_entity_resolution/src/data_inspection.py",
            "code/business_entity_resolution/src/matching_model.py",
        }

        with zipfile.ZipFile(created_zip, "r") as zf:
            archive_files = set(zf.namelist())
            for expected in expected_entries:
                self.assertIn(expected, archive_files, f"Missing required archive file: {expected}")
                info = zf.getinfo(expected)
                self.assertGreater(info.file_size, 0, f"Archive entry {expected} is empty (0 bytes)")

        # Cleanup test zip
        if zip_path.exists():
            zip_path.unlink()


if __name__ == "__main__":
    unittest.main()
