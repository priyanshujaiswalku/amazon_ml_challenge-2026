# Phase 1: Exploratory Data Analysis & Integrity Audit Report

**Amazon ML Challenge 2026 — Business Entity Resolution**

## 1. Executive Summary
This document records the audit findings across all raw competition data sources.
The inspection verifies schema integrity, delimiter precision, anomaly/null rates,
unique entity ID conventions, ground truth cardinality, and international geographic patterns.

## 2. Schema & Delimiter Verification (Task 1.1)

| File Name | Exists? | Rows | Columns | Delimiter | Ragged Rows | Schema Status |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| `train_source1.tsv` | YES | 10 | 4 | Tab (`\t`) | 0 | **PASSED** |
| `train_source2.tsv` | YES | 8 | 4 | Tab (`\t`) | 0 | **PASSED** |
| `train_source3.tsv` | YES | 6 | 4 | Tab (`\t`) | 0 | **PASSED** |
| `test_source1.tsv` | YES | 10 | 4 | Tab (`\t`) | 0 | **PASSED** |
| `test_source2.tsv` | YES | 5 | 4 | Tab (`\t`) | 0 | **PASSED** |
| `test_source3.tsv` | YES | 5 | 4 | Tab (`\t`) | 0 | **PASSED** |
| `train_ground_truth.tsv` | YES | 10 | 2 | Tab (`\t`) | 0 | **PASSED** |

## 3. Null & Anomaly Profiling (Task 1.2)

| File Name | Missing Name | Missing Address | Missing Country | Whitespace-only | Corrupt Chars |
|---|:---:|:---:|:---:|:---:|:---:|
| `train_source1.tsv` | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 | 0 |
| `train_source2.tsv` | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 | 0 |
| `train_source3.tsv` | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 | 0 |
| `test_source1.tsv` | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 | 0 |
| `test_source2.tsv` | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 | 0 |
| `test_source3.tsv` | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 | 0 |
| `train_ground_truth.tsv` | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 | 0 |

## 4. Entity ID Integrity & Uniqueness (Task 1.3)

| File Name | Total IDs | Unique IDs | Duplicate Count | Expected Prefix | Violations | Malformed IDs |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| `train_source1.tsv` | 10 | 10 | 0 | `S1-` | 0 | 0 |
| `train_source2.tsv` | 8 | 8 | 0 | `S2-` | 0 | 0 |
| `train_source3.tsv` | 6 | 6 | 0 | `S3-` | 0 | 0 |
| `test_source1.tsv` | 10 | 10 | 0 | `S1-` | 0 | 0 |
| `test_source2.tsv` | 5 | 5 | 0 | `S2-` | 0 | 0 |
| `test_source3.tsv` | 5 | 5 | 0 | `S3-` | 0 | 0 |

## 5. Ground Truth Cardinality Analysis (Task 1.4)

- **Total Reference Entities ($S_1$)**: 10
- **Singletons ($1 \to 0$)**: 2 (20.0%) — *Score 1.0 when empty*
- **One-to-One Matches ($1 \to 1$)**: 6 (60.0%)
- **One-to-Many Matches ($1 \to M$)**: 2 (20.0%)
- **Maximum Matches for Single $S_1$**: 2
- **Average Matches per $S_1$**: 1.0

### Source Breakdown
- **Matches from $S_2$ Only**: 4
- **Matches from $S_3$ Only**: 2
- **Matches from Both $S_2$ and $S_3$**: 2

### Invariant Compliance
- **Self-Match Violations ($S_1$ in matches)**: 0 (Must be 0)
- **Duplicate IDs in Comma-Separated List**: 0 (Must be 0)
- **Full $S_1$ Coverage**: PASS (100% matched)

## 6. Geographic Distribution & France Zero-Shot Profile (Task 1.5)

### Training vs Test Country Breakdown
- **Train Countries**: `{'US': 12, 'IN': 12}`
- **Test Countries**: `{'FR': 11, 'US': 5, 'IN': 4}`
- **Open-Set France Preserved**: YES (Verified zero-shot invariant)

### France Out-of-Distribution Profile
- **Total France Records**: 11
- **Accented Characters Detected**: 6
- **5-Digit Postal Code Conformance**: 10
- **French Street Types (rue, ave, blvd)**: 11
- **French Corporate Suffixes (sarl, sas, sa)**: 4

### Regional Signals for US & India
- **India PIN Matches**: 16 | **India Landmark Anchors**: 4
- **US ZIP Matches**: 15
