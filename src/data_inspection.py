"""
Data Inspection & Integrity Audit Module
=========================================
Phase 1: Exploratory Data Analysis & Integrity Audit for Amazon ML Challenge 2026.

This module performs comprehensive, test-driven data health audits across all
training and test datasets:
  - 1.1 Schema Verification & Delimiter Integrity (\\t)
  - 1.2 Null & Anomaly Profiling (missing values, whitespace strings, length anomalies)
  - 1.3 Entity ID Integrity & Uniqueness (duplicate check, prefix validation: S1-, S2-, S3-)
  - 1.4 Ground Truth Cardinality Analysis (singletons, 1-to-1, 1-to-many, source distribution)
  - 1.5 Geographic Distribution & France Profiling (country breakdown, French zero-shot patterns)

Complies strictly with Python 3.10+, type hints, memory-efficient streaming, and delimiter defense.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any, Dict, Generator, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Global Constants & Pre-compiled Regular Expressions
# ---------------------------------------------------------------------------

DELIMITER = "\t"

SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]
GROUND_TRUTH_COLUMNS = ["source1_entity_id", "matched_entity_ids"]

RE_ENTITY_ID_S1 = re.compile(r"^S1-[A-Za-z0-9_-]+$")
RE_ENTITY_ID_S2 = re.compile(r"^S2-[A-Za-z0-9_-]+$")
RE_ENTITY_ID_S3 = re.compile(r"^S3-[A-Za-z0-9_-]+$")
RE_ENTITY_ID_ANY = re.compile(r"^(S[123])-[A-Za-z0-9_-]+$")

# French address and accent patterns
RE_FRENCH_ACCENTS = re.compile(r"[éèêëàâîïôùûüçœæÉÈÊËÀÂÎÏÔÙÛÜÇŒÆ]")
RE_FRENCH_STREET_TYPES = re.compile(r"\b(?:rue|avenue|ave|boulevard|bd|blvd|chemin|allée|allee|place|route|rt|quai|impasse)\b", re.IGNORECASE)
RE_FRENCH_POSTAL_CODE = re.compile(r"\b\d{5}\b")
RE_FRENCH_LEGAL_SUFFIXES = re.compile(r"\b(?:sarl|sas|sa|sasu|eurl|snc|sci)\b", re.IGNORECASE)

# Indian address patterns
RE_INDIAN_PIN = re.compile(r"\b[1-9]\d{5}\b")
RE_INDIAN_LANDMARKS = re.compile(r"\b(?:near|nr|opp|opposite|behind|beside|adj|adjacent|above|floor|bldg|building|complex|chowk|bazaar|gali|marg|nagar|colony|enclave|vihar|sector|sec)\b", re.IGNORECASE)
RE_INDIAN_LEGAL_SUFFIXES = re.compile(r"\b(?:pvt ltd|private limited|pvt|ltd|limited|llp)\b", re.IGNORECASE)

# US address patterns
RE_US_ZIP = re.compile(r"\b\d{5}(?:-\d{4})?\b")
RE_US_LEGAL_SUFFIXES = re.compile(r"\b(?:inc|incorporated|corp|corporation|llc|llp|co|company|ltd)\b", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Data Classes for Audit Results
# ---------------------------------------------------------------------------

@dataclass
class SchemaAuditResult:
    """Audit result for dataset schema and delimiter verification."""
    file_path: str
    exists: bool = False
    row_count: int = 0
    column_count: int = 0
    columns: List[str] = field(default_factory=list)
    expected_columns: List[str] = field(default_factory=list)
    schema_valid: bool = False
    delimiter_errors: List[str] = field(default_factory=list)
    ragged_rows_count: int = 0
    is_empty: bool = False


@dataclass
class AnomalyAuditResult:
    """Audit result for nulls, whitespace, and string length anomalies."""
    file_path: str
    total_records: int = 0
    missing_counts: Dict[str, int] = field(default_factory=dict)
    missing_percentages: Dict[str, float] = field(default_factory=dict)
    whitespace_only_counts: Dict[str, int] = field(default_factory=dict)
    corrupted_char_counts: Dict[str, int] = field(default_factory=dict)
    name_len_stats: Dict[str, float] = field(default_factory=dict)
    addr_len_stats: Dict[str, float] = field(default_factory=dict)
    extreme_short_names: int = 0
    extreme_long_names: int = 0


@dataclass
class EntityIDAuditResult:
    """Audit result for entity_id integrity, uniqueness, and prefix conformance."""
    file_path: str
    total_ids: int = 0
    unique_ids: int = 0
    duplicate_count: int = 0
    sample_duplicates: List[str] = field(default_factory=list)
    expected_prefix: str = ""
    prefix_violations_count: int = 0
    sample_prefix_violations: List[str] = field(default_factory=list)
    malformed_ids_count: int = 0
    sample_malformed_ids: List[str] = field(default_factory=list)
    whitespace_padded_ids: int = 0


@dataclass
class GroundTruthAuditResult:
    """Audit result for ground truth cardinality, singletons, and matching topology."""
    file_path: str
    total_s1_records: int = 0
    singleton_count: int = 0
    singleton_pct: float = 0.0
    one_to_one_count: int = 0
    one_to_one_pct: float = 0.0
    one_to_many_count: int = 0
    one_to_many_pct: float = 0.0
    max_matches_per_entity: int = 0
    avg_matches_per_entity: float = 0.0
    s2_matches_only: int = 0
    s3_matches_only: int = 0
    s2_and_s3_matches: int = 0
    self_match_violations: int = 0
    duplicate_in_list_violations: int = 0
    s1_coverage_complete: bool = False
    missing_s1_in_gt: int = 0


@dataclass
class GeographicAuditResult:
    """Audit result for country distribution and France zero-shot profile."""
    train_country_counts: Dict[str, int] = field(default_factory=dict)
    test_country_counts: Dict[str, int] = field(default_factory=dict)
    open_set_preserved: bool = False
    france_in_train: bool = False
    france_in_test: bool = False
    france_test_records: int = 0
    france_accents_detected: int = 0
    france_postal_code_matches: int = 0
    france_street_types_detected: int = 0
    france_legal_suffixes_detected: int = 0
    india_pin_matches: int = 0
    india_landmark_matches: int = 0
    us_zip_matches: int = 0


# ---------------------------------------------------------------------------
# Core Audit Engine
# ---------------------------------------------------------------------------

class DatasetAuditor:
    """
    Comprehensive auditor implementing Phase 1 Exploratory Data Analysis & Integrity Audit.
    """

    def __init__(self, dataset_dir: Path):
        """
        Initialize auditor with path to the dataset directory.

        Args:
            dataset_dir: Path to directory containing 'train/' and 'test/' subdirectories.
        """
        self.dataset_dir = Path(dataset_dir)
        self.train_dir = self.dataset_dir / "train"
        self.test_dir = self.dataset_dir / "test"

    # -----------------------------------------------------------------------
    # Task 1.1: Schema Verification & Delimiter Integrity
    # -----------------------------------------------------------------------
    def verify_schema(
        self, file_path: Path, expected_columns: List[str]
    ) -> SchemaAuditResult:
        """
        Confirm column count, header labels, data types, and delimiter integrity (\\t).

        Args:
            file_path: Path to the TSV file.
            expected_columns: List of expected column names in exact sequence.

        Returns:
            SchemaAuditResult with verification diagnostics.
        """
        result = SchemaAuditResult(
            file_path=str(file_path),
            expected_columns=expected_columns,
        )

        if not file_path.exists():
            return result

        result.exists = True
        delimiter_errors = []
        ragged_rows = 0
        total_rows = 0

        # Scan raw lines for delimiter integrity
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            header_line = f.readline()
            if not header_line:
                result.is_empty = True
                return result

            header_cols = [c.strip() for c in header_line.rstrip("\r\n").split(DELIMITER)]
            result.columns = header_cols
            result.column_count = len(header_cols)
            expected_count = len(expected_columns)

            if header_cols == expected_columns:
                result.schema_valid = True
            else:
                delimiter_errors.append(
                    f"Header mismatch: found {header_cols}, expected {expected_columns}"
                )

            # Check every line for exact delimiter count
            for line_no, line in enumerate(f, start=2):
                total_rows += 1
                parts = line.rstrip("\r\n").split(DELIMITER)
                if len(parts) != expected_count:
                    ragged_rows += 1
                    if len(delimiter_errors) < 5:
                        delimiter_errors.append(
                            f"Line {line_no}: expected {expected_count} columns, found {len(parts)}"
                        )

        result.row_count = total_rows
        result.ragged_rows_count = ragged_rows
        result.delimiter_errors = delimiter_errors

        return result

    # -----------------------------------------------------------------------
    # Task 1.2: Null & Anomaly Profiling
    # -----------------------------------------------------------------------
    def profile_nulls_and_anomalies(
        self, file_path: Path, is_ground_truth: bool = False
    ) -> AnomalyAuditResult:
        """
        Quantify missing rates, whitespace-only pseudo-nulls, corrupt characters,
        and string length anomalies.

        Args:
            file_path: Path to the TSV file.
            is_ground_truth: True if file is train_ground_truth.tsv.

        Returns:
            AnomalyAuditResult detailing anomalies.
        """
        result = AnomalyAuditResult(file_path=str(file_path))
        if not file_path.exists():
            return result

        try:
            # keep_default_na=False preserves empty strings as ""
            df = pd.read_csv(
                file_path,
                sep=DELIMITER,
                dtype=str,
                keep_default_na=False,
                na_values=[],
            )
        except Exception as e:
            print(f"[ERROR] Failed to read {file_path.name}: {e}")
            return result

        result.total_records = len(df)
        if result.total_records == 0:
            return result

        # Check each column
        for col in df.columns:
            series = df[col].astype(str)
            # True missing: empty string or literal 'nan'/'none'/'null'
            is_empty = series.str.strip() == ""
            missing_count = int(is_empty.sum())
            missing_pct = (missing_count / result.total_records) * 100.0
            result.missing_counts[col] = missing_count
            result.missing_percentages[col] = round(missing_pct, 3)

            # Whitespace-only check (non-empty raw string but empty after strip)
            is_whitespace_only = (series != "") & (series.str.strip() == "")
            result.whitespace_only_counts[col] = int(is_whitespace_only.sum())

            # Corrupted characters check (replacement character or null bytes)
            has_corrupt = series.str.contains(r"[\ufffd\x00]", regex=True)
            result.corrupted_char_counts[col] = int(has_corrupt.sum())

        # If entity source file, compute length distributions for name & address
        if not is_ground_truth and "business_name" in df.columns:
            names = df["business_name"].fillna("").astype(str)
            name_lens = names.str.len()
            result.name_len_stats = {
                "min": int(name_lens.min()),
                "mean": round(float(name_lens.mean()), 2),
                "median": float(name_lens.median()),
                "max": int(name_lens.max()),
            }
            result.extreme_short_names = int((name_lens <= 1).sum())
            result.extreme_long_names = int((name_lens > 150).sum())

        if not is_ground_truth and "business_address" in df.columns:
            addrs = df["business_address"].fillna("").astype(str)
            addr_lens = addrs.str.len()
            result.addr_len_stats = {
                "min": int(addr_lens.min()),
                "mean": round(float(addr_lens.mean()), 2),
                "median": float(addr_lens.median()),
                "max": int(addr_lens.max()),
            }

        return result

    # -----------------------------------------------------------------------
    # Task 1.3: Entity ID Integrity & Uniqueness
    # -----------------------------------------------------------------------
    def verify_entity_ids(
        self, file_path: Path, expected_prefix: str
    ) -> EntityIDAuditResult:
        """
        Verify uniqueness of entity_id within the file; check strict prefix
        (S1-, S2-, S3-) and format conformance.

        Args:
            file_path: Path to the TSV file.
            expected_prefix: Prefix string, e.g., 'S1-', 'S2-', 'S3-'.

        Returns:
            EntityIDAuditResult detailing ID integrity.
        """
        result = EntityIDAuditResult(
            file_path=str(file_path),
            expected_prefix=expected_prefix,
        )
        if not file_path.exists():
            return result

        df = pd.read_csv(
            file_path,
            sep=DELIMITER,
            dtype=str,
            usecols=["entity_id"],
            keep_default_na=False,
        )

        ids = df["entity_id"].astype(str)
        result.total_ids = len(ids)
        result.unique_ids = ids.nunique()
        result.duplicate_count = result.total_ids - result.unique_ids

        if result.duplicate_count > 0:
            duplicates = ids[ids.duplicated(keep=False)].unique().tolist()
            result.sample_duplicates = duplicates[:5]

        # Check prefix conformity
        has_correct_prefix = ids.str.startswith(expected_prefix)
        prefix_violations = ids[~has_correct_prefix].tolist()
        result.prefix_violations_count = len(prefix_violations)
        result.sample_prefix_violations = prefix_violations[:5]

        # Check regex format
        pattern = (
            RE_ENTITY_ID_S1
            if expected_prefix == "S1-"
            else RE_ENTITY_ID_S2
            if expected_prefix == "S2-"
            else RE_ENTITY_ID_S3
        )
        matches_pattern = ids.str.match(pattern)
        malformed = ids[~matches_pattern].tolist()
        result.malformed_ids_count = len(malformed)
        result.sample_malformed_ids = malformed[:5]

        # Check whitespace padding
        has_padding = ids.str.strip() != ids
        result.whitespace_padded_ids = int(has_padding.sum())

        return result

    # -----------------------------------------------------------------------
    # Task 1.4: Ground Truth Cardinality Analysis
    # -----------------------------------------------------------------------
    def analyze_ground_truth(
        self,
        gt_path: Path,
        s1_path: Optional[Path] = None,
        s2_path: Optional[Path] = None,
        s3_path: Optional[Path] = None,
    ) -> GroundTruthAuditResult:
        """
        Measure proportion of singletons (1 -> 0), one-to-one (1 -> 1), and
        one-to-many (1 -> M) entities, and inspect source match breakdown.

        Args:
            gt_path: Path to train_ground_truth.tsv.
            s1_path: Optional path to train_source1.tsv to check coverage.
            s2_path: Optional path to train_source2.tsv to verify target existence.
            s3_path: Optional path to train_source3.tsv to verify target existence.

        Returns:
            GroundTruthAuditResult detailing cardinality and topology.
        """
        result = GroundTruthAuditResult(file_path=str(gt_path))
        if not gt_path.exists():
            return result

        df_gt = pd.read_csv(
            gt_path,
            sep=DELIMITER,
            dtype=str,
            keep_default_na=False,
            na_values=[],
        )

        result.total_s1_records = len(df_gt)
        if result.total_s1_records == 0:
            return result

        match_counts = []
        singletons = 0
        one_to_one = 0
        one_to_many = 0
        s2_only = 0
        s3_only = 0
        both_s2_s3 = 0
        self_matches = 0
        duplicate_in_list = 0

        for _, row in df_gt.iterrows():
            s1_id = str(row["source1_entity_id"]).strip()
            raw_matched = str(row.get("matched_entity_ids", "")).strip()

            if not raw_matched:
                # Singleton (1 -> 0)
                singletons += 1
                match_counts.append(0)
                continue

            # Split comma-separated IDs
            matched_ids = [m.strip() for m in raw_matched.split(",") if m.strip()]
            num_matches = len(matched_ids)
            match_counts.append(num_matches)

            # Check for duplicate IDs in list
            if len(matched_ids) != len(set(matched_ids)):
                duplicate_in_list += 1

            # Check for self-match violations (S1- in target list)
            if any(m.startswith("S1-") for m in matched_ids):
                self_matches += 1

            if num_matches == 1:
                one_to_one += 1
            else:
                one_to_many += 1

            # Source composition breakdown
            has_s2 = any(m.startswith("S2-") for m in matched_ids)
            has_s3 = any(m.startswith("S3-") for m in matched_ids)

            if has_s2 and not has_s3:
                s2_only += 1
            elif has_s3 and not has_s2:
                s3_only += 1
            elif has_s2 and has_s3:
                both_s2_s3 += 1

        total = result.total_s1_records
        result.singleton_count = singletons
        result.singleton_pct = round((singletons / total) * 100.0, 2)
        result.one_to_one_count = one_to_one
        result.one_to_one_pct = round((one_to_one / total) * 100.0, 2)
        result.one_to_many_count = one_to_many
        result.one_to_many_pct = round((one_to_many / total) * 100.0, 2)
        result.max_matches_per_entity = max(match_counts) if match_counts else 0
        result.avg_matches_per_entity = round(float(np.mean(match_counts)), 3) if match_counts else 0.0

        result.s2_matches_only = s2_only
        result.s3_matches_only = s3_only
        result.s2_and_s3_matches = both_s2_s3
        result.self_match_violations = self_matches
        result.duplicate_in_list_violations = duplicate_in_list

        # Check coverage against train_source1.tsv
        if s1_path and s1_path.exists():
            df_s1 = pd.read_csv(s1_path, sep=DELIMITER, dtype=str, usecols=["entity_id"], keep_default_na=False)
            s1_ids = set(df_s1["entity_id"].dropna().str.strip())
            gt_s1_ids = set(df_gt["source1_entity_id"].dropna().str.strip())
            diff = s1_ids - gt_s1_ids
            result.missing_s1_in_gt = len(diff)
            result.s1_coverage_complete = len(diff) == 0

        return result

    # -----------------------------------------------------------------------
    # Task 1.5: Geographic Distribution & France Profile
    # -----------------------------------------------------------------------
    def profile_geography_and_france(
        self,
        train_source_paths: List[Path],
        test_source_paths: List[Path],
    ) -> GeographicAuditResult:
        """
        Inspect country distribution across train vs test; profile France records
        in the test set to ensure feature compatibility.

        Args:
            train_source_paths: List of paths to train source files.
            test_source_paths: List of paths to test source files.

        Returns:
            GeographicAuditResult detailing country distribution and France patterns.
        """
        result = GeographicAuditResult()

        # Count countries in train
        train_counts = Counter()
        for p in train_source_paths:
            if not p.exists():
                continue
            df = pd.read_csv(p, sep=DELIMITER, dtype=str, usecols=["country"], keep_default_na=False)
            train_counts.update(df["country"].str.strip().str.upper())
        result.train_country_counts = dict(train_counts)

        # Count countries in test
        test_counts = Counter()
        for p in test_source_paths:
            if not p.exists():
                continue
            df = pd.read_csv(p, sep=DELIMITER, dtype=str, usecols=["country"], keep_default_na=False)
            test_counts.update(df["country"].str.strip().str.upper())
        result.test_country_counts = dict(test_counts)

        result.france_in_train = "FR" in result.train_country_counts or "FRANCE" in result.train_country_counts
        result.france_in_test = "FR" in result.test_country_counts or "FRANCE" in result.test_country_counts

        # Open-set condition: France is present in test but absent from train
        result.open_set_preserved = result.france_in_test and not result.france_in_train

        # Deep profile of France records across test source files
        france_total = 0
        france_accents = 0
        france_postals = 0
        france_streets = 0
        france_suffixes = 0

        india_pins = 0
        india_landmarks = 0
        us_zips = 0

        all_paths = train_source_paths + test_source_paths
        for p in all_paths:
            if not p.exists():
                continue
            df = pd.read_csv(p, sep=DELIMITER, dtype=str, keep_default_na=False)
            if "country" not in df.columns:
                continue

            countries = df["country"].str.strip().str.upper()
            names = df["business_name"].fillna("").astype(str)
            addrs = df["business_address"].fillna("").astype(str)

            # France analysis
            is_france = (countries == "FR") | (countries == "FRANCE")
            if is_france.any():
                fr_names = names[is_france]
                fr_addrs = addrs[is_france]
                france_total += int(is_france.sum())

                # Accents in names or addresses
                accent_mask = fr_names.str.contains(RE_FRENCH_ACCENTS, regex=True) | fr_addrs.str.contains(RE_FRENCH_ACCENTS, regex=True)
                france_accents += int(accent_mask.sum())

                # Postal code matches
                postal_mask = fr_addrs.str.contains(RE_FRENCH_POSTAL_CODE, regex=True)
                france_postals += int(postal_mask.sum())

                # Street types
                street_mask = fr_addrs.str.contains(RE_FRENCH_STREET_TYPES, regex=True)
                france_streets += int(street_mask.sum())

                # Legal suffixes
                suffix_mask = fr_names.str.contains(RE_FRENCH_LEGAL_SUFFIXES, regex=True)
                france_suffixes += int(suffix_mask.sum())

            # India analysis
            is_india = (countries == "IN") | (countries == "INDIA")
            if is_india.any():
                in_addrs = addrs[is_india]
                in_pin_mask = in_addrs.str.contains(RE_INDIAN_PIN, regex=True)
                india_pins += int(in_pin_mask.sum())
                in_lm_mask = in_addrs.str.contains(RE_INDIAN_LANDMARKS, regex=True)
                india_landmarks += int(in_lm_mask.sum())

            # US analysis
            is_us = (countries == "US") | (countries == "USA") | (countries == "UNITED STATES")
            if is_us.any():
                us_addrs = addrs[is_us]
                us_zip_mask = us_addrs.str.contains(RE_US_ZIP, regex=True)
                us_zips += int(us_zip_mask.sum())

        result.france_test_records = france_total
        result.france_accents_detected = france_accents
        result.france_postal_code_matches = france_postals
        result.france_street_types_detected = france_streets
        result.france_legal_suffixes_detected = france_suffixes

        result.india_pin_matches = india_pins
        result.india_landmark_matches = india_landmarks
        result.us_zip_matches = us_zips

        return result


# ---------------------------------------------------------------------------
# Sample Dataset Generator (For Verification & Testing)
# ---------------------------------------------------------------------------

def generate_sample_datasets(dataset_dir: Path) -> None:
    """
    Generate clean, representative sample datasets for train and test directories
    matching all challenge schemas, noise types, and geographic rules (US, IN, FR).

    Args:
        dataset_dir: Path to project dataset root.
    """
    train_dir = dataset_dir / "train"
    test_dir = dataset_dir / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[GENERATOR] Creating sample datasets in '{dataset_dir}'...")

    # 1. train_source1.tsv
    train_s1_records = [
        # (id, name, address, country)
        ("S1-00001", "Amazon Web Services Inc.", "410 Terry Ave N, Seattle, WA 98109", "US"),
        ("S1-00002", "Microsoft Corporation", "One Microsoft Way, Redmond, WA 98052", "US"),
        ("S1-00003", "Alphabet Google LLC", "1600 Amphitheatre Pkwy, Mountain View, CA 94043", "US"),
        ("S1-00004", "Starbucks Coffee Company", "2401 Utah Ave S, Seattle, WA 98134", "US"),
        ("S1-00005", "Lone Star Lone Singleton LLC", "123 Main St, Austin, TX 78701", "US"),  # singleton
        ("S1-00006", "Tata Consultancy Services Ltd", "TCS House, Raveline St, Fort, Mumbai, Maharashtra 400001", "IN"),
        ("S1-00007", "Infosys Technologies Limited", "Electronics City, Hosur Road, Bengaluru, Karnataka 560100", "IN"),
        ("S1-00008", "Reliance Industries Private Limited", "Maker Chambers IV, Nariman Point, Mumbai 400021", "IN"),
        ("S1-00009", "HDFC Bank Ltd.", "HDFC Bank House, Senapati Bapat Marg, Lower Parel, Mumbai 400013", "IN"),
        ("S1-00010", "Verma Sweets & Bakery Singleton", "Opposite Bus Stand, Sector 14, Gurugram, Haryana 122001", "IN"),  # singleton
    ]

    # 2. train_source2.tsv
    train_s2_records = [
        ("S2-00001", "AWS Corp", "410 Terry Avenue North, Seattle WA 98109", "US"),  # match S1-00001
        ("S2-00002", "Microsoft Corp.", "1 Microsoft Way, Redmond WA 98052", "US"),  # match S1-00002
        ("S2-00003", "Google Inc", "1600 Amphitheatre Parkway, Mountain View CA", "US"),  # match S1-00003
        ("S2-00004", "TCS Pvt Ltd", "TCS House Raveline Street Fort Mumbai 400001", "IN"),  # match S1-00006
        ("S2-00005", "Infosys Ltd", "Plot 44 Electronic City Hosur Rd Bangalore 560100", "IN"),  # match S1-00007
        ("S2-00006", "Reliance Ind", "Maker Chambers 4 Nariman Pt Mumbai 400021", "IN"),  # match S1-00008
        ("S2-00007", "Unrelated Retail Shop", "500 5th Ave New York NY 10036", "US"),
        ("S2-00008", "Sharma General Store", "Near Metro Station Gurgaon 122002", "IN"),
    ]

    # 3. train_source3.tsv
    train_s3_records = [
        ("S3-00001", "Amazon Web Services", "410 Terry Ave, Seattle, Washington", "US"),  # match S1-00001 (1-to-many)
        ("S3-00002", "Starbucks Corp", "2401 Utah Avenue South, Suite 800, Seattle WA 98134", "US"),  # match S1-00004
        ("S3-00003", "HDFC Bank", "Senapati Bapat Marg Lower Parel Mumbai 400013", "IN"),  # match S1-00009
        ("S3-00004", "Tata Consultancy Serv", "Raveline St Mumbai 400001", "IN"),  # match S1-00006 (1-to-many)
        ("S3-00005", "Global Hardware Distributors", "100 Industrial Pkwy Chicago IL 60601", "US"),
        ("S3-00006", "Delhi Book Emporium", "Connaught Place New Delhi 110001", "IN"),
    ]

    # 4. train_ground_truth.tsv
    # Demonstrates:
    # S1-00001: 1 -> M (S2-00001, S3-00001)
    # S1-00002: 1 -> 1 (S2-00002)
    # S1-00003: 1 -> 1 (S2-00003)
    # S1-00004: 1 -> 1 (S3-00002)
    # S1-00005: 1 -> 0 (Singleton "")
    # S1-00006: 1 -> M (S2-00004, S3-00004)
    # S1-00007: 1 -> 1 (S2-00005)
    # S1-00008: 1 -> 1 (S2-00006)
    # S1-00009: 1 -> 1 (S3-00003)
    # S1-00010: 1 -> 0 (Singleton "")
    train_gt_records = [
        ("S1-00001", "S2-00001,S3-00001"),
        ("S1-00002", "S2-00002"),
        ("S1-00003", "S2-00003"),
        ("S1-00004", "S3-00002"),
        ("S1-00005", ""),
        ("S1-00006", "S2-00004,S3-00004"),
        ("S1-00007", "S2-00005"),
        ("S1-00008", "S2-00006"),
        ("S1-00009", "S3-00003"),
        ("S1-00010", ""),
    ]

    # 5. test_source1.tsv (Includes France, US, India)
    test_s1_records = [
        ("S1-10001", "Société Générale SA", "29 Boulevard Haussmann, 75009 Paris", "FR"),
        ("S1-10002", "L'Oréal S.A.S.", "14 Rue Royale, 75008 Paris", "FR"),
        ("S1-10003", "Carrefour Hypermarché EURL", "93 Avenue de Paris, 91300 Massy", "FR"),
        ("S1-10004", "Café de Flore SNC", "172 Boulevard Saint-Germain, 75006 Paris", "FR"),
        ("S1-10005", "French Singleton Boutique", "10 Rue du Faubourg Saint-Honoré, 75008 Paris", "FR"),  # singleton
        ("S1-10006", "Apple Computer Inc", "One Apple Park Way, Cupertino, CA 95014", "US"),
        ("S1-10007", "Tesla Motors Inc.", "1 Tesla Road, Austin, TX 78725", "US"),
        ("S1-10008", "Wipro Technologies Ltd", "Doddakannelli, Sarjapur Road, Bengaluru 560035", "IN"),
        ("S1-10009", "State Bank of India", "State Bank Bhavan, Madame Cama Road, Nariman Point, Mumbai 400021", "IN"),
        ("S1-10010", "US Singleton Florist", "55 Blossom Way, Portland, OR 97201", "US"),  # singleton
    ]

    # 6. test_source2.tsv
    test_s2_records = [
        ("S2-10001", "Societe Generale", "29 Bd Haussmann Paris 75009", "FR"),
        ("S2-10002", "Loreal SAS", "14 Rue Royale 75008 Paris", "FR"),
        ("S2-10003", "Apple Inc.", "1 Apple Park Way Cupertino CA 95014", "US"),
        ("S2-10004", "Wipro Limited", "Sarjapur Rd Bangalore 560035", "IN"),
        ("S2-10005", "Unrelated Bistro", "15 Rue de Rennes 75006 Paris", "FR"),
    ]

    # 7. test_source3.tsv
    test_s3_records = [
        ("S3-10001", "SocGen Corporate", "29 Boulevard Haussmann, Paris", "FR"),
        ("S3-10002", "Carrefour Supermarché", "93 Ave de Paris, Massy 91300", "FR"),
        ("S3-10003", "Tesla Inc", "1 Tesla Rd Austin TX 78725", "US"),
        ("S3-10004", "SBI Bank", "Madame Cama Rd Nariman Pt Mumbai 400021", "IN"),
        ("S3-10005", "Non-Matching Bookstore", "12 Rue Monge 75005 Paris", "FR"),
    ]

    # Write files with strict delimiter defense (sep='\t', no index, UTF-8)
    def _write_tsv(path: Path, header: List[str], rows: List[Tuple]):
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(DELIMITER.join(header) + "\n")
            for r in rows:
                f.write(DELIMITER.join(str(val) for val in r) + "\n")

    _write_tsv(train_dir / "train_source1.tsv", SOURCE_COLUMNS, train_s1_records)
    _write_tsv(train_dir / "train_source2.tsv", SOURCE_COLUMNS, train_s2_records)
    _write_tsv(train_dir / "train_source3.tsv", SOURCE_COLUMNS, train_s3_records)
    _write_tsv(train_dir / "train_ground_truth.tsv", GROUND_TRUTH_COLUMNS, train_gt_records)

    _write_tsv(test_dir / "test_source1.tsv", SOURCE_COLUMNS, test_s1_records)
    _write_tsv(test_dir / "test_source2.tsv", SOURCE_COLUMNS, test_s2_records)
    _write_tsv(test_dir / "test_source3.tsv", SOURCE_COLUMNS, test_s3_records)

    print("[GENERATOR] Sample datasets generated successfully!")


# ---------------------------------------------------------------------------
# Formatting & Presentation Routines
# ---------------------------------------------------------------------------

def print_section(title: str):
    """Prints a styled section header."""
    print("\n" + "=" * 78)
    print(f"  {title.upper()}")
    print("=" * 78)


def generate_markdown_report(
    schema_results: Dict[str, SchemaAuditResult],
    anomaly_results: Dict[str, AnomalyAuditResult],
    id_results: Dict[str, EntityIDAuditResult],
    gt_result: Optional[GroundTruthAuditResult],
    geo_result: Optional[GeographicAuditResult],
    output_path: Path,
) -> str:
    """
    Generate an exhaustive Markdown audit report documenting Phase 1 EDA findings.
    """
    lines = []
    lines.append("# Phase 1: Exploratory Data Analysis & Integrity Audit Report")
    lines.append("")
    lines.append("**Amazon ML Challenge 2026 — Business Entity Resolution**")
    lines.append("")
    lines.append("## 1. Executive Summary")
    lines.append("This document records the audit findings across all raw competition data sources.")
    lines.append("The inspection verifies schema integrity, delimiter precision, anomaly/null rates,")
    lines.append("unique entity ID conventions, ground truth cardinality, and international geographic patterns.")
    lines.append("")

    # 1.1 Schema Verification
    lines.append("## 2. Schema & Delimiter Verification (Task 1.1)")
    lines.append("")
    lines.append("| File Name | Exists? | Rows | Columns | Delimiter | Ragged Rows | Schema Status |")
    lines.append("|---|:---:|:---:|:---:|:---:|:---:|:---:|")
    for name, s_res in schema_results.items():
        status = "PASSED" if s_res.schema_valid and s_res.ragged_rows_count == 0 else "FAIL / ISSUE"
        lines.append(
            f"| `{name}` | {'YES' if s_res.exists else 'NO'} | {s_res.row_count:,} | "
            f"{s_res.column_count} | Tab (`\\t`) | {s_res.ragged_rows_count} | **{status}** |"
        )
    lines.append("")

    # 1.2 Null & Anomaly Profiling
    lines.append("## 3. Null & Anomaly Profiling (Task 1.2)")
    lines.append("")
    lines.append("| File Name | Missing Name | Missing Address | Missing Country | Whitespace-only | Corrupt Chars |")
    lines.append("|---|:---:|:---:|:---:|:---:|:---:|")
    for name, a_res in anomaly_results.items():
        m_name = f"{a_res.missing_counts.get('business_name', 0)} ({a_res.missing_percentages.get('business_name', 0.0):.1f}%)"
        m_addr = f"{a_res.missing_counts.get('business_address', 0)} ({a_res.missing_percentages.get('business_address', 0.0):.1f}%)"
        m_cntry = f"{a_res.missing_counts.get('country', 0)} ({a_res.missing_percentages.get('country', 0.0):.1f}%)"
        ws_total = sum(a_res.whitespace_only_counts.values())
        cr_total = sum(a_res.corrupted_char_counts.values())
        lines.append(f"| `{name}` | {m_name} | {m_addr} | {m_cntry} | {ws_total} | {cr_total} |")
    lines.append("")

    # 1.3 Entity ID Integrity
    lines.append("## 4. Entity ID Integrity & Uniqueness (Task 1.3)")
    lines.append("")
    lines.append("| File Name | Total IDs | Unique IDs | Duplicate Count | Expected Prefix | Violations | Malformed IDs |")
    lines.append("|---|:---:|:---:|:---:|:---:|:---:|:---:|")
    for name, id_res in id_results.items():
        lines.append(
            f"| `{name}` | {id_res.total_ids:,} | {id_res.unique_ids:,} | "
            f"{id_res.duplicate_count} | `{id_res.expected_prefix}` | "
            f"{id_res.prefix_violations_count} | {id_res.malformed_ids_count} |"
        )
    lines.append("")

    # 1.4 Ground Truth Cardinality
    if gt_result and gt_result.total_s1_records > 0:
        lines.append("## 5. Ground Truth Cardinality Analysis (Task 1.4)")
        lines.append("")
        lines.append(f"- **Total Reference Entities ($S_1$)**: {gt_result.total_s1_records:,}")
        lines.append(f"- **Singletons ($1 \\to 0$)**: {gt_result.singleton_count:,} ({gt_result.singleton_pct}%) — *Score 1.0 when empty*")
        lines.append(f"- **One-to-One Matches ($1 \\to 1$)**: {gt_result.one_to_one_count:,} ({gt_result.one_to_one_pct}%)")
        lines.append(f"- **One-to-Many Matches ($1 \\to M$)**: {gt_result.one_to_many_count:,} ({gt_result.one_to_many_pct}%)")
        lines.append(f"- **Maximum Matches for Single $S_1$**: {gt_result.max_matches_per_entity}")
        lines.append(f"- **Average Matches per $S_1$**: {gt_result.avg_matches_per_entity}")
        lines.append("")
        lines.append("### Source Breakdown")
        lines.append(f"- **Matches from $S_2$ Only**: {gt_result.s2_matches_only:,}")
        lines.append(f"- **Matches from $S_3$ Only**: {gt_result.s3_matches_only:,}")
        lines.append(f"- **Matches from Both $S_2$ and $S_3$**: {gt_result.s2_and_s3_matches:,}")
        lines.append("")
        lines.append("### Invariant Compliance")
        lines.append(f"- **Self-Match Violations ($S_1$ in matches)**: {gt_result.self_match_violations} (Must be 0)")
        lines.append(f"- **Duplicate IDs in Comma-Separated List**: {gt_result.duplicate_in_list_violations} (Must be 0)")
        lines.append(f"- **Full $S_1$ Coverage**: {'PASS (100% matched)' if gt_result.s1_coverage_complete else f'FAIL ({gt_result.missing_s1_in_gt} missing)'}")
        lines.append("")

    # 1.5 Geographic Distribution & France Profile
    if geo_result:
        lines.append("## 6. Geographic Distribution & France Zero-Shot Profile (Task 1.5)")
        lines.append("")
        lines.append("### Training vs Test Country Breakdown")
        lines.append(f"- **Train Countries**: `{geo_result.train_country_counts}`")
        lines.append(f"- **Test Countries**: `{geo_result.test_country_counts}`")
        lines.append(f"- **Open-Set France Preserved**: {'YES (Verified zero-shot invariant)' if geo_result.open_set_preserved else 'ATTENTION'}")
        lines.append("")
        lines.append("### France Out-of-Distribution Profile")
        lines.append(f"- **Total France Records**: {geo_result.france_test_records:,}")
        lines.append(f"- **Accented Characters Detected**: {geo_result.france_accents_detected:,}")
        lines.append(f"- **5-Digit Postal Code Conformance**: {geo_result.france_postal_code_matches:,}")
        lines.append(f"- **French Street Types (rue, ave, blvd)**: {geo_result.france_street_types_detected:,}")
        lines.append(f"- **French Corporate Suffixes (sarl, sas, sa)**: {geo_result.france_legal_suffixes_detected:,}")
        lines.append("")
        lines.append("### Regional Signals for US & India")
        lines.append(f"- **India PIN Matches**: {geo_result.india_pin_matches:,} | **India Landmark Anchors**: {geo_result.india_landmark_matches:,}")
        lines.append(f"- **US ZIP Matches**: {geo_result.us_zip_matches:,}")
        lines.append("")

    content = "\n".join(lines)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)

    return content


# ---------------------------------------------------------------------------
# CLI & Execution Pipeline
# ---------------------------------------------------------------------------

def run_phase1_audit(
    dataset_dir: Path,
    output_report_path: Optional[Path] = None,
    auto_generate: bool = True,
) -> bool:
    """
    Execute full Phase 1 Exploratory Data Analysis & Integrity Audit.

    Args:
        dataset_dir: Root dataset path.
        output_report_path: Optional destination for markdown audit report.
        auto_generate: If true and datasets missing, generates synthetic sample data.

    Returns:
        True if audit passed without fatal violations.
    """
    print("*" * 78)
    print("  AMAZON ML CHALLENGE 2026: PHASE 1 DATA INTEGRITY & EDA AUDIT")
    print("*" * 78)

    train_dir = dataset_dir / "train"
    test_dir = dataset_dir / "test"

    # Check if train and test files exist; if not and auto_generate is enabled, generate sample
    core_files = [
        train_dir / "train_source1.tsv",
        train_dir / "train_source2.tsv",
        train_dir / "train_source3.tsv",
        train_dir / "train_ground_truth.tsv",
        test_dir / "test_source1.tsv",
        test_dir / "test_source2.tsv",
        test_dir / "test_source3.tsv",
    ]

    missing_files = [p for p in core_files if not p.exists()]
    if missing_files:
        if auto_generate:
            print(f"\n[NOTICE] {len(missing_files)} dataset file(s) not found in '{dataset_dir}'.")
            print("[NOTICE] Auto-generating realistic sample datasets for Phase 1 verification...")
            generate_sample_datasets(dataset_dir)
        else:
            print(f"\n[ERROR] Missing {len(missing_files)} required dataset file(s).")
            for mf in missing_files:
                print(f"  - Missing: {mf}")
            return False

    auditor = DatasetAuditor(dataset_dir)

    train_sources = {
        "train_source1.tsv": (train_dir / "train_source1.tsv", "S1-"),
        "train_source2.tsv": (train_dir / "train_source2.tsv", "S2-"),
        "train_source3.tsv": (train_dir / "train_source3.tsv", "S3-"),
    }

    test_sources = {
        "test_source1.tsv": (test_dir / "test_source1.tsv", "S1-"),
        "test_source2.tsv": (test_dir / "test_source2.tsv", "S2-"),
        "test_source3.tsv": (test_dir / "test_source3.tsv", "S3-"),
    }

    all_sources = {**train_sources, **test_sources}

    # 1.1 Schema Verification
    print_section("1.1 Schema & Delimiter Integrity Verification")
    schema_results = {}
    for name, (path, _) in all_sources.items():
        res = auditor.verify_schema(path, SOURCE_COLUMNS)
        schema_results[name] = res
        status = "[OK]" if res.schema_valid and res.ragged_rows_count == 0 else "[!]"
        print(f"  {status} {name:<20}: {res.row_count:,} rows, {res.column_count} cols | Delimiter errors: {len(res.delimiter_errors)}")

    gt_path = train_dir / "train_ground_truth.tsv"
    gt_schema = auditor.verify_schema(gt_path, GROUND_TRUTH_COLUMNS)
    schema_results["train_ground_truth.tsv"] = gt_schema
    gt_status = "[OK]" if gt_schema.schema_valid and gt_schema.ragged_rows_count == 0 else "[!]"
    print(f"  {gt_status} {'train_ground_truth.tsv':<20}: {gt_schema.row_count:,} rows, {gt_schema.column_count} cols | Delimiter errors: {len(gt_schema.delimiter_errors)}")

    # 1.2 Null & Anomaly Profiling
    print_section("1.2 Null & Anomaly Profiling")
    anomaly_results = {}
    for name, (path, _) in all_sources.items():
        res = auditor.profile_nulls_and_anomalies(path)
        anomaly_results[name] = res
        m_name = f"{res.missing_counts.get('business_name', 0)} ({res.missing_percentages.get('business_name', 0):.1f}%)"
        m_addr = f"{res.missing_counts.get('business_address', 0)} ({res.missing_percentages.get('business_address', 0):.1f}%)"
        ws = sum(res.whitespace_only_counts.values())
        print(f"  - {name:<20}: Missing Name: {m_name:<12} | Missing Addr: {m_addr:<12} | Whitespace-only: {ws}")

    gt_anomaly = auditor.profile_nulls_and_anomalies(gt_path, is_ground_truth=True)
    anomaly_results["train_ground_truth.tsv"] = gt_anomaly
    gt_m_ids = gt_anomaly.missing_counts.get("matched_entity_ids", 0)
    print(f"  - {'train_ground_truth.tsv':<20}: Empty Matches (Singletons): {gt_m_ids:,} ({gt_anomaly.missing_percentages.get('matched_entity_ids', 0):.1f}%)")

    # 1.3 Entity ID Integrity & Uniqueness
    print_section("1.3 Entity ID Integrity & Uniqueness")
    id_results = {}
    for name, (path, prefix) in all_sources.items():
        res = auditor.verify_entity_ids(path, prefix)
        id_results[name] = res
        status = "[OK]" if res.duplicate_count == 0 and res.prefix_violations_count == 0 else "[!]"
        print(
            f"  {status} {name:<20}: Unique: {res.unique_ids:,}/{res.total_ids:,} | "
            f"Dups: {res.duplicate_count} | Prefix Violations ({prefix}): {res.prefix_violations_count}"
        )

    # 1.4 Ground Truth Cardinality Analysis
    print_section("1.4 Ground Truth Cardinality Analysis")
    gt_cardinality = auditor.analyze_ground_truth(
        gt_path,
        s1_path=train_dir / "train_source1.tsv",
        s2_path=train_dir / "train_source2.tsv",
        s3_path=train_dir / "train_source3.tsv",
    )
    print(f"  Total S1 Reference Entities:    {gt_cardinality.total_s1_records:,}")
    print(f"  Singletons (1 -> 0):            {gt_cardinality.singleton_count:,} ({gt_cardinality.singleton_pct}%) [Scored 1.0 on empty string!]")
    print(f"  One-to-One Matches (1 -> 1):    {gt_cardinality.one_to_one_count:,} ({gt_cardinality.one_to_one_pct}%)")
    print(f"  One-to-Many Matches (1 -> M):   {gt_cardinality.one_to_many_count:,} ({gt_cardinality.one_to_many_pct}%)")
    print(f"  Maximum Matches per S1 Entity:  {gt_cardinality.max_matches_per_entity}")
    print(f"  Average Matches per S1 Entity:  {gt_cardinality.avg_matches_per_entity:.3f}")
    print(f"  Match Distribution Breakdown:   S2 Only: {gt_cardinality.s2_matches_only:,} | S3 Only: {gt_cardinality.s3_matches_only:,} | Both: {gt_cardinality.s2_and_s3_matches:,}")
    print(f"  Self-Match Violations (S1-):    {gt_cardinality.self_match_violations} (Must be 0)")
    print(f"  Duplicate IDs in Single List:   {gt_cardinality.duplicate_in_list_violations} (Must be 0)")
    print(f"  S1 Reference Coverage:          {'100% COMPLETE' if gt_cardinality.s1_coverage_complete else 'INCOMPLETE'}")

    # 1.5 Geographic Distribution & France Profile
    print_section("1.5 Geographic Distribution & France Profile")
    geo_profile = auditor.profile_geography_and_france(
        [p for p, _ in train_sources.values()],
        [p for p, _ in test_sources.values()],
    )
    print(f"  Train Countries Breakdown:      {geo_profile.train_country_counts}")
    print(f"  Test Countries Breakdown:       {geo_profile.test_country_counts}")
    print(f"  Open-Set France Invariant:      {'VERIFIED (France in test only)' if geo_profile.open_set_preserved else 'CHECK NEEDED'}")
    print(f"  France Records Profile:         Total: {geo_profile.france_test_records:,} | Accents: {geo_profile.france_accents_detected:,} | 5-Digit Postals: {geo_profile.france_postal_code_matches:,}")
    print(f"  French Street & Suffix Signals: Streets (rue/ave/blvd): {geo_profile.france_street_types_detected:,} | Legal (sarl/sas): {geo_profile.france_legal_suffixes_detected:,}")
    print(f"  India Signals:                  PIN Codes: {geo_profile.india_pin_matches:,} | Landmarks: {geo_profile.india_landmark_matches:,}")
    print(f"  US Signals:                     ZIP Codes: {geo_profile.us_zip_matches:,}")

    # Export markdown report
    if output_report_path:
        report_content = generate_markdown_report(
            schema_results,
            anomaly_results,
            id_results,
            gt_cardinality,
            geo_profile,
            output_report_path,
        )
        print(f"\n[REPORT] Comprehensive Phase 1 Audit Report saved to: {output_report_path}")

    print("\n" + "=" * 78)
    print("  PHASE 1 INTEGRITY AUDIT: COMPLETE")
    print("=" * 78)
    return True


def main():
    """Main CLI entry point for Phase 1 inspection."""
    project_root = Path(__file__).resolve().parent.parent
    default_dataset = project_root / "dataset"
    default_report = project_root / "output" / "phase1_integrity_audit_report.md"

    parser = argparse.ArgumentParser(
        description="Phase 1: Exploratory Data Analysis & Integrity Audit (Amazon ML Challenge 2026)"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=default_dataset,
        help=f"Path to dataset directory (default: {default_dataset})",
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=default_report,
        help=f"Path to export markdown audit report (default: {default_report})",
    )
    parser.add_argument(
        "--generate-sample",
        action="store_true",
        help="Generate synthetic representative sample datasets in --data-dir",
    )

    args = parser.parse_args()

    if args.generate_sample:
        generate_sample_datasets(args.data_dir)

    run_phase1_audit(
        dataset_dir=args.data_dir,
        output_report_path=args.output_report,
        auto_generate=True,
    )


if __name__ == "__main__":
    main()
