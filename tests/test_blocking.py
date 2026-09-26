"""
Unit Tests for Phase 3: High-Recall Multi-Key Inverted Index Blocking
======================================================================
Tests multi-key signature generation, inverted index postings, country isolation,
top-K candidate budget constraints, and candidate recall benchmarks.
"""

import unittest
from pathlib import Path
import pandas as pd

from src.blocking import (
    BlockingEngine,
    extract_blocking_keys,
    run_benchmark,
    PROJECT_ROOT,
)


class TestBlockingPhase3(unittest.TestCase):
    """Test suite for Phase 3 Multi-Key Blocking Engine."""

    def test_extract_blocking_keys_us(self):
        """Test blocking keys generation on a structured US entity."""
        keys = extract_blocking_keys(
            clean_name="amazon web services inc",
            core_name="amazon web services",
            clean_addr="410 terry ave n seattle wa 98109",
            country="US",
        )
        # Should contain Core Name Key, First Word, 2-Word, Prefix, and Postal/Addr
        self.assertIn("US#CN#amazon web services", keys)
        self.assertIn("US#1W#amazon", keys)
        self.assertIn("US#2W#amazon_web", keys)
        self.assertIn("US#PFX4#amaz", keys)
        self.assertIn("US#PIN_NAME#98109_amazon", keys)

    def test_extract_blocking_keys_france(self):
        """Test blocking keys generation on French zero-shot entity."""
        keys = extract_blocking_keys(
            clean_name="societe generale sa",
            core_name="societe generale",
            clean_addr="29 blvd haussmann 75009 paris",
            country="FR",
        )
        self.assertIn("FR#CN#societe generale", keys)
        self.assertIn("FR#1W#societe", keys)
        self.assertIn("FR#2W#societe_generale", keys)
        self.assertIn("FR#PFX4#soci", keys)
        self.assertIn("FR#PIN_NAME#75009_societe", keys)

    def test_country_isolation_invariant(self):
        """Verify strict country partitioning: cross-border matches must never occur."""
        engine = BlockingEngine(max_candidates_per_entity=10)

        # Index US and Indian records with similar core names
        engine.index_record("S2-US01", "Reliance Retail Inc", "123 Main St", "US")
        engine.index_record("S2-IN01", "Reliance Retail Pvt Ltd", "Nariman Point Mumbai", "IN")

        # Query from US should only retrieve the US candidate
        us_cands = engine.retrieve_candidates_for_query("Reliance Retail", "123 Main St", "US")
        self.assertIn("S2-US01", us_cands)
        self.assertNotIn("S2-IN01", us_cands)

        # Query from India should only retrieve the Indian candidate
        in_cands = engine.retrieve_candidates_for_query("Reliance Retail", "Mumbai", "IN")
        self.assertIn("S2-IN01", in_cands)
        self.assertNotIn("S2-US01", in_cands)

    def test_top_k_candidate_capping(self):
        """Verify candidate list never exceeds max_candidates budget."""
        k = 3
        engine = BlockingEngine(max_candidates_per_entity=k)

        # Index 10 targets sharing the same core name
        for i in range(10):
            engine.index_record(f"S2-{i:03d}", "Starbucks Coffee", f"{i} Street Ave", "US")

        cands = engine.retrieve_candidates_for_query("Starbucks Coffee", "1 Street Ave", "US")
        self.assertLessEqual(len(cands), k)

    def test_candidate_recall_on_training_data(self):
        """Verify that multi-key blocking achieves >= 95% candidate recall on train ground truth."""
        recall = run_benchmark(sample_size=10, top_k=25)
        self.assertGreaterEqual(recall, 95.0, f"Candidate recall ({recall:.2f}%) must be >= 95.0%")


if __name__ == "__main__":
    unittest.main()
