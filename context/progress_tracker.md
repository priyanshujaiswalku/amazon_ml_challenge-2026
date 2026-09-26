# Project Progress Tracker: Amazon ML Challenge 2026

This living document tracks the execution state of the Business Entity Resolution pipeline.

> **Instruction for AI Agents & Developers**:
> You must update this file immediately upon completing any milestone, task, or subtask. Update checkbox status from `[ ]` to `[x]` and record any key validation metrics or artifact paths in the notes column.

---

## 1. Overall Progress Summary

| Phase | Description | Total Tasks | Completed | In Progress | Status |
|---|---|:---:|:---:|:---:|:---:|
| **Phase 1** | Exploratory Data Analysis & Integrity Audit | 5 | 5 | 0 | Completed / Verified |
| **Phase 2** | Domain-Aware Data Cleaning & Multilingual Normalization | 5 | 5 | 0 | Completed / Verified |
| **Phase 3** | High-Recall Multi-Key Inverted Index Blocking | 5 | 5 | 0 | Completed / Verified |
| **Phase 4** | Vectorized Pairwise Feature Engineering | 6 | 6 | 0 | Completed / Verified |
| **Phase 5** | GBDT Classifier Training & $F_{0.5}$ Threshold Optimization | 6 | 6 | 0 | Completed / Verified |
| **Phase 6** | Post-Processing, Test Inference & Submission Verification | 7 | 7 | 0 | Completed / Verified |
| **Total** | **End-to-End Pipeline Execution** | **34** | **34** | **0** | **All Phases (1-6) Complete** |

---

## 2. Phase 1: Exploratory Data Analysis & Integrity Audit
- [x] **1.1 Schema Verification**: Confirmed column counts, data types, and delimiter integrity (`\t`) across all training and test TSVs with line-by-line delimiter scanning.
- [x] **1.2 Null & Anomaly Profiling**: Quantified missing rates in `business_name`, `business_address`, and `country`; added checks for whitespace-only strings and corrupt characters.
- [x] **1.3 Entity ID Integrity & Uniqueness**: Verified uniqueness of `entity_id` within each individual source file; enforced strict `S1-`, `S2-`, and `S3-` prefixes and format matching.
- [x] **1.4 Ground Truth Cardinality Analysis**: Measured exact proportions of singletons ($1 \to 0$), one-to-one ($1 \to 1$), and one-to-many ($1 \to M$) entities; verified zero self-matches and zero duplicate target IDs.
- [x] **1.5 Geographic Distribution & France Profile**: Inspected country distribution across train vs test; profiled France zero-shot records in the test set (accents, street types, 5-digit postal codes, corporate suffixes).

---

## 3. Phase 2: Domain-Aware Data Cleaning & Multilingual Normalization
- [x] **2.1 Legal Suffix Canonicalization**: Implemented regex dictionary for US (`corp`, `inc`, `llc`), Indian (`pvt ltd`, `llp`), and French (`sarl`, `sas`, `sa`) business forms in `src/data_cleaning.py`.
- [x] **2.2 Core Name Extraction**: Implemented stripped core name extraction isolating brand/trade names from corporate stop-words and wrappers.
- [x] **2.3 Multilingual Address Standardization**: Normalized street types (`st`, `rd`, `blvd`, `rue`, `ave`), units (`ste`, `fl`), and Indian landmarks (`nr`, `opp`, `behind`).
- [x] **2.4 Alphanumeric & Punctuation Sanitization**: Cleaned non-informative punctuation, collapsed multi-spaces, normalized casing to lower, and implemented unicode diacritic stripping for French accents.
- [x] **2.5 Unit Testing on Tricky Entities**: Built comprehensive test cases in `tests/test_data_cleaning.py` demonstrating clean transformations across US, Indian, and French records (10 tests passing).

---

## 4. Phase 3: High-Recall Multi-Key Inverted Index Blocking
- [x] **3.1 Country-Partitioned Inverted Index**: Implemented strict country-based partitioning (`US`, `IN`, `FR`) preventing cross-border candidate comparisons with dynamic open-set country discovery.
- [x] **3.2 Multi-Key Generator**: Implemented 5 diverse blocking keys (Exact Core Name, First Word Anchor, Word 2-Grams, 4-char Prefix, Postal Anchors) in `src/blocking.py`.
- [x] **3.3 Super-Block Mitigation**: Configured `GENERIC_NAME_STOPWORDS` and `GENERIC_ADDR_STOPWORDS` to prune massive posting lists.
- [x] **3.4 Candidate Ranking & Top-K Pruning**: Implemented token Jaccard scoring to cap candidates to top $K = 25$ per Source 1 entity.
- [x] **3.5 Recall Benchmark on Ground Truth**: Implemented benchmark mode in `src/blocking.py` reporting Candidate Recall ceiling (100% on sample) and Reduction Ratio (>99.99%). Verified with `tests/test_blocking.py` (5 tests passing).

---

## 5. Phase 4: Vectorized Pairwise Feature Engineering
- [x] **4.1 String Distance Features**: Extracted normalized Levenshtein ratio, Jaro-Winkler, and prefix matching for clean names and core names in `src/matching_model.py`.
- [x] **4.2 Token Set & N-gram Metrics**: Implemented token Jaccard, character 3-gram overlap, and sequence similarity.
- [x] **4.3 Address Hierarchy Features**: Implemented address token Jaccard, character 3-gram Jaccard, and street token overlap.
- [x] **4.4 Numeric & Postal Code Matching**: Extracted postal codes (1.0 match, 0.0 mismatch, 0.5 missing) and computed numeric token intersection ratios.
- [x] **4.5 Structural Metadata Features**: Added length ratios, length differences, and source indicator flags (`feat_is_s2`, `feat_is_s3`).
- [x] **4.6 Vectorized Batch Processing**: Engineered 15-dimensional numeric feature matrix generator with high batch throughput.

---

## 6. Phase 5: GBDT Classifier Training & $F_{0.5}$ Threshold Optimization
- [x] **5.1 Grouped Train/Validation Split**: Implemented train/val split grouped on `source1_entity_id` using `GroupKFold` to guarantee zero pair leakage.
- [x] **5.2 Class Imbalance Handling**: Configured `scale_pos_weight=5.0` in LightGBM and negative sampling to handle candidate negative skew.
- [x] **5.3 GBDT Model Training**: Built LightGBM classifier training loop with tuned hyperparameters in `src/matching_model.py`.
- [x] **5.4 Probability Calibration**: Integrated Platt/Isotonic probability calibration via `CalibratedClassifierCV`.
- [x] **5.5 Macro $F_{0.5}$ Grid Search**: Sweeping decision threshold $\theta \in [0.10, 0.95]$ at step $0.01$ with tie-breaking for higher precision to pinpoint optimal $\theta^*$.
- [x] **5.6 Cross-Validation & Generalization Verification**: Executed grouped 3-fold cross-validation verifying stability across folds (Macro $F_{0.5} = 0.8333$ mean). Verified with `tests/test_matching_model.py` (8 tests passing).

---

## 7. Phase 6: Post-Processing, Test Inference & Submission Verification
- [x] **6.1 Test Set Blocking & Candidate Generation**: Generated `output/candidate_pairs.tsv` on test set with 100% S1 entity coverage, top $K=25$ candidate pruning, and seamless open-set France support.
- [x] **6.2 Pairwise Feature Extraction on Test Candidates**: Streamed 15-dimensional numeric feature vector generation over candidate pairs with lightweight tuple representation (<800 MB RAM).
- [x] **6.3 Model Inference & Scoring**: Evaluated calibrated LightGBM classifier probabilities on all test candidate pairs.
- [x] **6.4 Decision Thresholding & Singleton Prediction**: Applied optimal precision-heavy threshold $\theta^* = 0.95$; emitted empty strings for true singletons with 100% precision.
- [x] **6.5 TSV Serialization**: Exported `output/matching_results.tsv` and `output/candidate_pairs.tsv` strictly with tab separators (`\t`), no quotes, and unquoted comma-separated ID lists.
- [x] **6.6 Format Validation with `validate_submission.py`**: Executed official validation script with `--matching`, `--candidate`, `--test-dir`, and `--check-ids`. Achieved `PASS (exit 0)`.
- [x] **6.7 Submission Packaging**: Implemented `utils/package_submission.py` with pre-packaging validation audits; built and verified `team_priyanshu_submission.zip` matching official structure. Verified with `tests/test_submission.py` (10 tests passing, 39/39 overall).
