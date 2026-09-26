"""
Unit Tests for Phase 1: Data Inspection & Integrity Audit
==========================================================
Tests the core validation, delimiter verification, anomaly profiling,
prefix checks, ground truth cardinality, and geographic profiling.
"""

from pathlib import Path
import tempfile
import unittest
import pandas as pd

from src.data_inspection import (
    DatasetAuditor,
    SchemaAuditResult,
    AnomalyAuditResult,
    EntityIDAuditResult,
    GroundTruthAuditResult,
    GeographicAuditResult,
    generate_sample_datasets,
    SOURCE_COLUMNS,
    GROUND_TRUTH_COLUMNS,
    DELIMITER,
)


class TestDataInspectionPhase1(unittest.TestCase):
    """Test suite for Phase 1 Data Inspection & Integrity Audit."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base_dir = Path(self.temp_dir.name)
        self.dataset_dir = self.base_dir / "dataset"
        self.auditor = DatasetAuditor(self.dataset_dir)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_sample_dataset_generation_and_schema(self):
        """Verify that sample dataset generator produces compliant TSV files."""
        generate_sample_datasets(self.dataset_dir)

        # Check train_source1.tsv schema
        s1_path = self.dataset_dir / "train" / "train_source1.tsv"
        schema_res = self.auditor.verify_schema(s1_path, SOURCE_COLUMNS)
        self.assertTrue(schema_res.exists)
        self.assertTrue(schema_res.schema_valid)
        self.assertEqual(schema_res.column_count, 4)
        self.assertEqual(schema_res.ragged_rows_count, 0)
        self.assertEqual(len(schema_res.delimiter_errors), 0)

        # Check train_ground_truth.tsv schema
        gt_path = self.dataset_dir / "train" / "train_ground_truth.tsv"
        gt_schema = self.auditor.verify_schema(gt_path, GROUND_TRUTH_COLUMNS)
        self.assertTrue(gt_schema.exists)
        self.assertTrue(gt_schema.schema_valid)
        self.assertEqual(gt_schema.column_count, 2)
        self.assertEqual(gt_schema.ragged_rows_count, 0)

    def test_ragged_delimiter_detection(self):
        """Test that ragged rows (inconsistent tabs) are accurately flagged."""
        corrupt_file = self.base_dir / "ragged.tsv"
        with open(corrupt_file, "w", encoding="utf-8") as f:
            f.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
            f.write("S1-001\tAcme Corp\t123 Main St\tUS\n")
            f.write("S1-002\tBroken Line With Extra Tab\tAddr\tExtra\tUS\n")  # 5 columns instead of 4

        res = self.auditor.verify_schema(corrupt_file, SOURCE_COLUMNS)
        self.assertEqual(res.ragged_rows_count, 1)
        self.assertFalse(res.schema_valid and res.ragged_rows_count == 0)

    def test_whitespace_and_null_profiling(self):
        """Test detection of nulls and whitespace-only strings."""
        test_file = self.base_dir / "nulls.tsv"
        with open(test_file, "w", encoding="utf-8") as f:
            f.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
            f.write("S1-001\t   \t123 Main St\tUS\n")  # whitespace only name
            f.write("S1-002\tReal Name\t\tUS\n")        # empty address
            f.write("S1-003\tAnother Name\t456 Oak Ave\tUS\n")

        res = self.auditor.profile_nulls_and_anomalies(test_file)
        self.assertEqual(res.total_records, 3)
        self.assertEqual(res.missing_counts["business_address"], 1)
        self.assertEqual(res.whitespace_only_counts["business_name"], 1)

    def test_entity_id_integrity_and_prefixes(self):
        """Test verification of S1, S2, S3 prefixes and duplicate detection."""
        test_file = self.base_dir / "ids.tsv"
        with open(test_file, "w", encoding="utf-8") as f:
            f.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
            f.write("S1-001\tName 1\tAddr 1\tUS\n")
            f.write("S1-002\tName 2\tAddr 2\tUS\n")
            f.write("S1-001\tName 1 Dupe\tAddr 1 Dupe\tUS\n")  # duplicate
            f.write("S2-999\tWrong Prefix\tAddr 3\tUS\n")     # wrong prefix for S1

        res = self.auditor.verify_entity_ids(test_file, expected_prefix="S1-")
        self.assertEqual(res.total_ids, 4)
        self.assertEqual(res.unique_ids, 3)
        self.assertEqual(res.duplicate_count, 1)
        self.assertEqual(res.prefix_violations_count, 1)
        self.assertIn("S2-999", res.sample_prefix_violations)

    def test_ground_truth_cardinality(self):
        """Test singleton identification and cardinality metrics in ground truth."""
        generate_sample_datasets(self.dataset_dir)
        gt_path = self.dataset_dir / "train" / "train_ground_truth.tsv"
        s1_path = self.dataset_dir / "train" / "train_source1.tsv"

        res = self.auditor.analyze_ground_truth(gt_path, s1_path=s1_path)
        self.assertEqual(res.total_s1_records, 10)
        self.assertEqual(res.singleton_count, 2)  # S1-00005 and S1-00010
        self.assertEqual(res.singleton_pct, 20.0)
        self.assertEqual(res.one_to_one_count, 6)
        self.assertEqual(res.one_to_many_count, 2)
        self.assertEqual(res.self_match_violations, 0)
        self.assertEqual(res.duplicate_in_list_violations, 0)
        self.assertTrue(res.s1_coverage_complete)

    def test_geographic_distribution_and_france_profile(self):
        """Test geographic distribution and France zero-shot profile."""
        generate_sample_datasets(self.dataset_dir)
        train_sources = [
            self.dataset_dir / "train" / "train_source1.tsv",
            self.dataset_dir / "train" / "train_source2.tsv",
            self.dataset_dir / "train" / "train_source3.tsv",
        ]
        test_sources = [
            self.dataset_dir / "test" / "test_source1.tsv",
            self.dataset_dir / "test" / "test_source2.tsv",
            self.dataset_dir / "test" / "test_source3.tsv",
        ]

        geo_res = self.auditor.profile_geography_and_france(train_sources, test_sources)
        self.assertNotIn("FR", geo_res.train_country_counts)
        self.assertIn("FR", geo_res.test_country_counts)
        self.assertTrue(geo_res.open_set_preserved)
        self.assertGreater(geo_res.france_test_records, 0)
        self.assertGreater(geo_res.france_accents_detected, 0)
        self.assertGreater(geo_res.france_postal_code_matches, 0)


if __name__ == "__main__":
    unittest.main()
