"""
Unit Tests for Phase 4 & Phase 5: Feature Extraction, GBDT Modeling, and F_0.5 Optimization
=============================================================================================
Tests string distance metrics, pairwise feature vector extraction, macro F_0.5 calculation
with singleton semantics, grouped cross-validation partitioning, and LightGBM model training.
"""

import unittest
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from src.matching_model import (
    FEATURE_COLUMNS,
    fast_levenshtein_ratio,
    fast_jaro_winkler,
    compute_jaccard,
    get_char_trigrams,
    extract_pair_features,
    calculate_macro_f05,
    optimize_decision_threshold,
    train_matching_model,
    run_cross_validation,
)


class TestMatchingModelPhase4And5(unittest.TestCase):
    """Test suite for Phase 4 & Phase 5 matching model pipeline."""

    def test_fast_levenshtein_ratio(self):
        """Test normalized Levenshtein similarity metric."""
        self.assertEqual(fast_levenshtein_ratio("apple", "apple"), 1.0)
        self.assertEqual(fast_levenshtein_ratio("", "apple"), 0.0)
        self.assertEqual(fast_levenshtein_ratio("apple", ""), 0.0)
        sim = fast_levenshtein_ratio("apple", "apply")
        self.assertAlmostEqual(sim, 0.8, places=2)

    def test_fast_jaro_winkler(self):
        """Test Jaro-Winkler string similarity with prefix bonus."""
        self.assertEqual(fast_jaro_winkler("google", "google"), 1.0)
        self.assertEqual(fast_jaro_winkler("", "google"), 0.0)
        jw = fast_jaro_winkler("martha", "marhta")
        self.assertGreater(jw, 0.90)

    def test_feature_vector_dimensions_and_semantics(self):
        """Verify feature extractor outputs exactly 15 features matching FEATURE_COLUMNS."""
        feats = extract_pair_features(
            s1_name="Amazon Web Services Inc.",
            s1_addr="410 Terry Ave N, Seattle, WA 98109",
            s1_country="US",
            target_id="S2-00001",
            target_name="AWS Corp",
            target_addr="410 Terry Avenue North, Seattle WA 98109",
            target_country="US",
        )
        self.assertEqual(len(feats), len(FEATURE_COLUMNS))
        self.assertEqual(len(feats), 15)

        feat_dict = dict(zip(FEATURE_COLUMNS, feats))

        # Check source signals
        self.assertEqual(feat_dict["feat_is_s2"], 1.0)
        self.assertEqual(feat_dict["feat_is_s3"], 0.0)

        # Check postal matching
        self.assertEqual(feat_dict["feat_postal_exact_match"], 1.0)

        # Check number overlap (both have 410 and 98109)
        self.assertGreater(feat_dict["feat_number_overlap_ratio"], 0.0)

        # Check Jaccard and Levenshtein values are bounded in [0.0, 1.0]
        for name, val in feat_dict.items():
            if name != "feat_name_length_diff":
                self.assertGreaterEqual(val, 0.0, f"{name} is below 0.0")
                self.assertLessEqual(val, 1.0, f"{name} is above 1.0")

    def test_postal_code_missing_and_mismatch(self):
        """Verify postal code exact match states: 1.0 match, 0.0 mismatch, 0.5 missing."""
        # Missing postal
        feats_missing = extract_pair_features(
            s1_name="Acme", s1_addr="Main St", s1_country="US",
            target_id="S2-1", target_name="Acme", target_addr="Oak Ave", target_country="US",
        )
        dict_missing = dict(zip(FEATURE_COLUMNS, feats_missing))
        self.assertEqual(dict_missing["feat_postal_exact_match"], 0.5)

        # Mismatch postal
        feats_mismatch = extract_pair_features(
            s1_name="Acme", s1_addr="123 Main St 98101", s1_country="US",
            target_id="S2-1", target_name="Acme", target_addr="123 Main St 75001", target_country="US",
        )
        dict_mismatch = dict(zip(FEATURE_COLUMNS, feats_mismatch))
        self.assertEqual(dict_mismatch["feat_postal_exact_match"], 0.0)

        # Match postal
        feats_match = extract_pair_features(
            s1_name="Acme", s1_addr="123 Main St 98101", s1_country="US",
            target_id="S2-1", target_name="Acme", target_addr="123 Main St 98101", target_country="US",
        )
        dict_match = dict(zip(FEATURE_COLUMNS, feats_match))
        self.assertEqual(dict_match["feat_postal_exact_match"], 1.0)

    def test_macro_f05_singleton_rules(self):
        """Verify Macro F_0.5 follows exact competition singleton evaluation rules."""
        # Case 1: Perfect singleton prediction (true is empty, predicted is empty) -> 1.0
        gt_singleton = {"S1-001": set()}
        pred_empty = {"S1-001": set()}
        f05, p, r = calculate_macro_f05(gt_singleton, pred_empty)
        self.assertEqual(f05, 1.0)
        self.assertEqual(p, 1.0)
        self.assertEqual(r, 1.0)

        # Case 2: Broken singleton prediction (true is empty, predicted is false match) -> 0.0
        pred_false_match = {"S1-001": {"S2-999"}}
        f05, p, r = calculate_macro_f05(gt_singleton, pred_false_match)
        self.assertEqual(f05, 0.0)
        self.assertEqual(p, 0.0)
        self.assertEqual(r, 0.0)

        # Case 3: True match with empty prediction (false negative) -> 0.0
        gt_match = {"S1-002": {"S2-001"}}
        f05, p, r = calculate_macro_f05(gt_match, pred_empty)
        self.assertEqual(f05, 0.0)

        # Case 4: True match with perfect prediction -> 1.0
        pred_exact = {"S1-002": {"S2-001"}}
        f05, p, r = calculate_macro_f05(gt_match, pred_exact)
        self.assertEqual(f05, 1.0)

        # Case 5: 1 TP, 1 FP on entity with 1 true match:
        # P = 1/2 = 0.5, R = 1/1 = 1.0
        # F_0.5 = 1.25 * 0.5 * 1.0 / (0.25 * 0.5 + 1.0) = 0.625 / 1.125 = 5/9 ~= 0.5556
        pred_imprecise = {"S1-002": {"S2-001", "S3-999"}}
        f05, p, r = calculate_macro_f05(gt_match, pred_imprecise)
        self.assertAlmostEqual(f05, 5.0 / 9.0, places=4)
        self.assertAlmostEqual(p, 0.5, places=4)
        self.assertAlmostEqual(r, 1.0, places=4)

    def test_grouped_kfold_no_entity_leakage(self):
        """Verify GroupKFold strictly prevents S1 entity ID leakage across folds."""
        entities = [f"S1-{i:03d}" for i in range(10)]
        # Simulate multiple pairs per entity
        s1_groups = np.array([e for e in entities for _ in range(3)])
        dummy_X = np.zeros((len(s1_groups), 2))
        dummy_y = np.zeros((len(s1_groups),))

        gkf = GroupKFold(n_splits=5)
        for train_idx, val_idx in gkf.split(dummy_X, dummy_y, groups=s1_groups):
            train_entities = set(s1_groups[train_idx])
            val_entities = set(s1_groups[val_idx])
            overlap = train_entities & val_entities
            self.assertEqual(len(overlap), 0, f"Entity leakage detected: {overlap}")

    def test_model_training_and_calibration_pipeline(self):
        """Verify that training pipeline trains LightGBM and saves model bundle."""
        model, threshold = train_matching_model(sample_size=10, top_k=25)
        self.assertIsNotNone(model)
        self.assertGreaterEqual(threshold, 0.10)
        self.assertLessEqual(threshold, 0.95)

    def test_cross_validation_execution(self):
        """Verify 5-fold cross-validation loop completes and reports valid metrics."""
        f05, p, r = run_cross_validation(sample_size=10, n_splits=3, top_k=25)
        self.assertGreaterEqual(f05, 0.0)
        self.assertLessEqual(f05, 1.0)


if __name__ == "__main__":
    unittest.main()
