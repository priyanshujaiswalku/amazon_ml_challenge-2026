# Experiment & Pipeline Registry: Amazon ML Challenge 2026

This living document catalogs reusable pipeline components, feature versions, candidate blocking strategies, and experimental evaluations across the project lifecycle.

---

## 1. Reusable Component Registry

| Component Name | File Location | Responsibility | Current Version | Status |
|---|---|---|---|---|
| **Data Inspector** | `src/data_inspection.py` | Dataset profiling, schema verification, null/anomaly detection, cardinality, France profiling | `v2.0` | Production Ready |
| **Multilingual Cleaner** | `src/data_cleaning.py` | Suffix stripping, address standardization for US/IN/FR | `v1.1` | Production Ready |
| **Multi-Key Blocker** | `src/blocking.py` | High-recall inverted index blocking with Jaccard ranking | `v1.2` | Production Ready |
| **Pairwise Feature Extractor** | `src/matching_model.py` | 15-feature vectorized pairwise similarity engine | `v1.0` | Production Ready |
| **LightGBM Classifier** | `src/matching_model.py` | Calibrated gradient boosted decision tree classifier | `v1.0` | Production Ready |
| **F_0.5 Threshold Optimizer** | `src/matching_model.py` | Macro F_0.5 decision threshold grid search engine | `v1.0` | Production Ready |
| **Submission Validator** | `utils/validate_submission.py` | Format, schema, and cross-file rule compliance verifier | `v1.0` | Official Utility |
| **Submission Packager** | `utils/package_submission.py` | Pre-validated ZIP archive packager adhering to competition structure | `v1.0` | Production Ready |

---

## 2. Feature Set Definitions

### Feature Set `v1.0` (15 Core Pairwise Features)
- **Names**: `feat_name_exact_core`, `feat_name_exact_clean`, `feat_name_token_jaccard`, `feat_name_char_trigram_jaccard`, `feat_name_levenshtein_ratio`, `feat_name_jaro_winkler`, `feat_name_length_diff`, `feat_name_length_ratio`.
- **Addresses**: `feat_addr_exact_clean`, `feat_addr_token_jaccard`, `feat_addr_char_trigram_jaccard`.
- **Postal & Numeric**: `feat_postal_exact_match`, `feat_number_overlap_ratio`.
- **Metadata**: `feat_is_s2`, `feat_is_s3`.

### Feature Set `v2.0` (Planned / Advanced Expansion)
- **TF-IDF Character N-grams**: Cosine similarity on character 2-4 grams across names.
- **Soundex / Double Metaphone**: Phonetic match indicators for Indian name transliterations.
- **Token Set Difference Ratio**: Asymmetric token overlap (detecting sub-brands vs parent brands).
- **Landmark Specific Match**: Boolean flag indicating match on extracted landmark keywords (`nr`, `opp`, `behind`).

---

## 3. Experiment & Benchmark Log

### Experiment: AUDIT-001 (Phase 1 Baseline Dataset Integrity & EDA Audit)
- **Date**: 2026-09-26
- **Objective**: Complete Phase 1 audit verifying schemas, tab delimiter integrity, null & anomaly distributions, entity ID prefix compliance, ground truth cardinality, and zero-shot France profiles.
- **Scope**: Evaluated `train_source1.tsv`, `train_source2.tsv`, `train_source3.tsv`, `train_ground_truth.tsv`, `test_source1.tsv`, `test_source2.tsv`, `test_source3.tsv`.
- **Results**:
  - **Schema & Delimiter Integrity**: 100% compliant with expected headers; 0 ragged rows; explicit tab (`\t`) delimiter verified.
  - **Null & Anomaly Rates**: 0 whitespace-only pseudo-null strings, 0 corrupt/replacement characters.
  - **Entity ID Integrity**: 100% unique IDs, 0 duplicate IDs, 0 prefix violations (`S1-`, `S2-`, `S3-`).
  - **Cardinality Breakdown**: 20% singletons ($1 \to 0$), 60% one-to-one ($1 \to 1$), 20% one-to-many ($1 \to M$). 0 self-match violations, 0 duplicate IDs in match lists.
  - **Geographic & France Zero-Shot Profile**: France (`FR`) verified present strictly in test set (absent from train); accented characters, 5-digit postals, and French street types (`rue`, `ave`, `blvd`) detected and profiled.
- **Artifacts**: `output/phase1_integrity_audit_report.md`
- **Takeaway**: Pipeline data contracts are completely verified and sound, establishing strict baseline integrity for Phase 2 data cleaning and Phase 3 blocking.

---

### Experiment: EXP-001 (Baseline Heuristic Blocking)
- **Date**: 2026-09-26
- **Objective**: Establish baseline candidate recall and reduction ratio using multi-key inverted index blocking.
- **Blocking Configuration**:
  - Keys: Exact Core Name, First Word Anchor, Word 2-Grams, 4-char Name Prefix, Postal + Word Anchor.
  - Candidate Cap: $K = 25$ per entity.
- **Results**:
  - **Candidate Recall**: `95.4%` on training ground truth.
  - **Reduction Ratio (RR)**: `> 99.98%` (Reduced 17.3T comparisons to ~25 per record).
- **Artifacts**: `output/candidate_pairs.tsv`
- **Takeaway**: Candidate recall exceeds the $95\%$ performance floor, proving blocking viability.

---

### Experiment: EXP-002 (LightGBM GBDT Pairwise Matcher + F_0.5 Optimization)
- **Date**: 2026-09-26
- **Objective**: Train pairwise binary classifier on Feature Set `v1.0` and optimize decision threshold for macro $F_{0.5}$.
- **Model Configuration**:
  - Algorithm: LightGBM Classifier (`num_leaves=31`, `max_depth=6`, `learning_rate=0.05`, `scale_pos_weight=5.0`).
  - Validation: 5-Fold Grouped Split on `source1_entity_id`.
  - Calibration: Isotonic Regression.
- **Threshold Optimization**:
  - Threshold swept from $0.10 \to 0.95$.
  - Optimal Threshold $\theta^* = 0.72$.
- **Validation Results**:
  - **Precision**: `0.842`
  - **Recall**: `0.781`
  - **Macro $F_{0.5}$ (incl. Singletons)**: `0.828`
- **Takeaway**: Setting $\theta^* = 0.72$ prioritizes precision, preventing false merges and heavily boosting singleton scores from $0.0 \to 1.0$.

---

### Experiment: EXP-004 (Phase 1-5 End-to-End Implementation & Cross-Validation)
- **Date**: 2026-09-26
- **Objective**: Implement and verify complete pipeline from Phase 1 through Phase 5 with unicode diacritic handling, dynamic open-set country blocking, 15 standard pairwise features, GroupKFold cross-validation, and CalibratedClassifierCV.
- **Key Pipeline Enhancements**:
  - **Phase 2 Cleaning**: Added unicode accent/diacritic stripping (`unicodedata.normalize('NFKD')`), dotted abbreviation collapse (`s.a.s.` -> `sas`), apostrophe brand contraction (`l'oréal` -> `loreal`), multi-word legal forms (`pvt ltd`, `sa`, `sarl`), and canonical open-set country mapping.
  - **Phase 3 Blocking**: Fixed hardcoded country partition bug by discovering unique countries dynamically from Source 1 (`df['country'].unique()`). Added super-block stopwords filtering, candidate ranking, and top-K ($K=25$) pruning.
  - **Phase 4 Features**: Extracted exact 15-dimensional feature vector with self-contained, high-performance normalized Levenshtein ratio and Jaro-Winkler string similarity metrics.
  - **Phase 5 Modeling**: Added negative pair sampling to ensure binary classifier stability. Implemented 5-fold `GroupKFold` on `source1_entity_id`, LightGBM training (`scale_pos_weight=5.0`), probability calibration (`CalibratedClassifierCV`), and precision-heavy Macro $F_{0.5}$ threshold sweep.
- **Validation Results**:
  - **Unit Tests**: 29/29 tests passing across all 4 test suites (`test_data_inspection.py`, `test_data_cleaning.py`, `test_blocking.py`, `test_matching_model.py`).
  - **Candidate Recall**: 100.0% on training sample; 0% cross-country bleed; open-set France/India/US entities all captured.
  - **Cross-Validation Macro $F_{0.5}$**: Mean $0.8333$ across folds with $\theta^* = 0.95$.
  - **Validator Compliance**: `utils/validate_submission.py` passed with exit code 0 (`PASS`) on all 10 test records with `--check-ids` enabled.
- **Artifacts**:
  - `output/matching_model.pkl` (Calibrated LightGBM model bundle)
  - `output/candidate_pairs.tsv` (10 test candidate sets)
  - `output/matching_results.tsv` (10 test match predictions)
- **Takeaway**: Pipeline is structurally sound, mathematically verified, fully compliant with contest constraints, and ready for Phase 6 full inference and packaging.

---

### Experiment: EXP-005 (Phase 6 Test Inference, Submissions Validation & Packaging Verification)
- **Date**: 2026-09-26
- **Objective**: Execute end-to-end inference on the test dataset across all three countries (including open-set France zero-shot records), verify candidate superset invariants, enforce tab delimiters and unquoted ID lists, achieve official validator PASS (exit 0), and automate compliant submission archive packaging.
- **Key Pipeline Deliverables**:
  - **Memory-Safe Test Inference Engine**: Upgraded `run_prediction_pipeline` in `src/matching_model.py` to use lightweight tuple representations `(business_name, business_address, country)` for target records, strictly operating well within memory budgets (<800 MB RAM).
  - **Invariant Assertions**: Implemented automated defensive checks in `run_prediction_pipeline` verifying 1:1 entity row matching with `test_source1.tsv`, zero duplicate rows, zero `S1-` self-matches, and strict mathematical candidate superset conformance ($\mathcal{M}_{\text{pred}}(e_1) \subseteq \mathcal{C}(e_1)$).
  - **Official Validation Gate**: Verified compliance with `utils/validate_submission.py` on both `output/matching_results.tsv` and `output/candidate_pairs.tsv` with `--check-ids` and `--test-dir dataset/test`. Result: `PASS — no blocking issues found. Safe to submit.`
  - **Automated Submission Packager**: Built `utils/package_submission.py` running pre-packaging validation audits and assembling `<team_name>_submission.zip` matching official contest archive structure (`output/`, `code/business_entity_resolution/`, `Documentation_template.md`).
  - **Phase 6 Verification Suite**: Created comprehensive test suite in `tests/test_submission.py` testing delimiters, headers, candidate superset invariants, singletons (`\t\n`), zero-shot France processing, validator execution, and zip archive manifest integrity.
- **Validation Results**:
  - **Unit Test Coverage**: 39/39 tests passing across all 5 test suites (10/10 in `test_submission.py`).
  - **Test Set Entity Coverage**: 10/10 Source 1 test entities processed (100% complete, 0 missing, 0 duplicates).
  - **Prediction Breakdown**: 7 entities with high-confidence matches, 3 singletons emitted as empty strings (`""`), 0 false merges on singletons.
  - **France Zero-Shot**: All 5 French test records (`S1-10001` through `S1-10005`) processed seamlessly without error or cross-border bleed.
  - **Validator Status**: Official `utils/validate_submission.py` returned Exit Code 0 (`PASS`).
  - **Submission Package**: Generated `team_priyanshu_submission.zip` (38.5 KB compressed, 130.5 KB uncompressed, 10 verified files).
- **Artifacts**:
  - `output/matching_results.tsv` (Test match predictions for leaderboard)
  - `output/candidate_pairs.tsv` (Test blocking candidate pairs)
  - `utils/package_submission.py` (Submission packager utility)
  - `tests/test_submission.py` (Phase 6 verification test suite)
  - `team_priyanshu_submission.zip` (Official competition submission package)
- **Takeaway**: Phase 6 is complete, fully verified, and ready for deployment. The submission package strictly fulfills all formatting, data integrity, and contest rules.

---

## 4. Instructions for Logging Future Experiments
Whenever you modify blocking keys, engineer new features, retrain models, or adjust thresholds:
1. Assign an experiment ID (e.g., `EXP-004`).
2. Document the exact changes and rationale.
3. Record validation metrics: Candidate Recall, Reduction Ratio, Precision, Recall, and Macro $F_{0.5}$.
4. Note artifact paths in `output/`.
5. Update `context/progress_tracker.md` to reflect the milestone status.

### Experiment: EXP-006 (Leakage-Free Calibration Evaluation)
- **Date**: 2026-09-27
- **Objective**: Remove validation-label leakage from probability calibration so threshold selection and reported validation metrics reflect an independent held-out fold.
- **Change**: Replaced isotonic calibration fitted on the validation fold with sigmoid calibration fitted by 3-fold CV within the training fold, in both grouped CV and the single holdout training path. Validation entity IDs are taken from the held-out pair groups so entities with no retrieved candidates are scored as empty predictions.
- **Expected benefit**: A more reliable operating threshold and less optimistic validation estimates; any leaderboard improvement must be measured after retraining and resubmission.
- **Benchmark status**: The grouped CV benchmark for this code change was not run. The dataset was later hydrated from Git LFS and a separate corrected holdout diagnostic is recorded under EXP-008.
- **Artifact**: `src/matching_model.py`
- **Takeaway**: Evaluation protocol is corrected; the previous reported validation score is not directly comparable because its calibrator was fitted on validation labels.

### Experiment: EXP-007 (Full-Test Model Inference)
- **Date**: 2026-09-27
- **Objective**: Produce complete leaderboard predictions from the full 1,732,544-row test Source 1 set within the submission window.
- **Inference strategy**: Desktop repository's disk-backed LightGBM pipeline, with country isolation and exact-core, prefix, or postal retrieval up to its configured candidate cap.
- **Results**: Official validator PASS. `matching_results.tsv`: 1,732,544 rows, 1,434,839 non-empty, 297,705 empty. `candidate_pairs.tsv`: 1,732,544 rows, 1,731,299 non-empty, 1,245 empty.
- **Leaderboard result**: User reports 0.466 after uploading this output. No test-label score can be computed locally.
- **Validation caveat**: The reported 0.9826 training validation is optimistic because the trainer forcibly adds each known true ID to candidates and fits calibration on the same validation labels. EXP-008 replaces this evaluation protocol.
- **Artifacts**: `output/matching_results.tsv`, `output/candidate_pairs.tsv`.
- **Takeaway**: Output completeness and format pass; the inference quality did not improve the reported leaderboard score.

### Experiment: EXP-008 (Unleaked Grouped Holdout Diagnostic)
- **Date**: 2026-09-27
- **Objective**: Diagnose the gap between the reported training validation score and the unchanged leaderboard score.
- **Protocol correction**: Split Source 1 rows before pair construction; do not append known true IDs or extra distractors to validation candidates; score all held-out Source 1 rows, including rows with no candidates; choose the threshold on held-out raw model probabilities.
- **Sample results**: 3,000 sampled S1 rows; 43,962 candidate pairs; 9,853 positives; sampled-index blocking recall `0.9444`; held-out macro F0.5 `0.9581`, precision `0.9805`, recall `0.9096`, threshold `0.87`.
- **Interpretation**: This is still a small sample and uses a sampled target index, so it is not a reliable prediction of leaderboard performance. The prior `0.9826` score was invalid because known positives were forcibly added to candidate sets and calibration was fitted on validation labels.
- **Current submission**: Desktop artifact copied to `output/matching_results.tsv`; 1,732,544 rows; official validator PASS. Its leaderboard result remains the only test-set performance evidence.
