# Project Build Plan: Amazon ML Challenge 2026

The project development lifecycle is partitioned into six sequential, test-driven phases. Each phase establishes a verified capability required by subsequent layers.

---

## Phase 1: Exploratory Data Analysis & Integrity Audit
**Objective**: Characterize the statistical distributions, missingness patterns, country proportions, and singleton rates across all training and test files.

- [ ] **1.1 Schema Verification**: Confirm column counts, data types, and delimiter integrity (`\t`) across `train_source1.tsv`, `train_source2.tsv`, `train_source3.tsv`, and `train_ground_truth.tsv`.
- [ ] **1.2 Null & Anomaly Profiling**: Quantify missing rates in `business_name`, `business_address`, and `country`. Check for whitespace-only strings or corrupt records.
- [ ] **1.3 Entity ID Integrity & Uniqueness**: Verify uniqueness of `entity_id` within each individual source file; ensure strict `S1-`, `S2-`, and `S3-` prefixes.
- [ ] **1.4 Ground Truth Cardinality Analysis**: Measure the exact proportion of singletons ($1 \to 0$), one-to-one ($1 \to 1$), and one-to-many ($1 \to M$) entities.
- [ ] **1.5 Geographic Distribution & France Profile**: Inspect country distribution across train vs test; profile France records in the test set to ensure feature compatibility.

---

## Phase 2: Domain-Aware Data Cleaning & Multilingual Normalization
**Objective**: Transform noisy, unstructured business names and addresses into clean canonical representations without destroying discriminative tokens.

- [ ] **2.1 Legal Suffix Canonicalization**: Build and test regex substitutions for US (`corp`, `inc`, `llc`, `ltd`), Indian (`pvt ltd`, `llp`, `co`), and French (`sarl`, `sas`, `sa`, `sasu`, `eurl`, `snc`) business forms.
- [ ] **2.2 Core Name Extraction**: Create a stripped "core name" representation isolating the primary brand/trade name from corporate wrappers and common stopwords.
- [ ] **2.3 Multilingual Address Standardization**: Normalize street types (`street` $\to$ `st`, `road` $\to$ `rd`, `boulevard` $\to$ `blvd`, `rue` $\to$ `rue`, `avenue` $\to$ `ave`), unit identifiers (`suite` $\to$ `ste`, `floor` $\to$ `fl`), and landmark indicators (`near` $\to$ `nr`, `opposite` $\to$ `opp`).
- [ ] **2.4 Alphanumeric & Punctuation Sanitization**: Clean non-informative punctuation, normalize repeated whitespaces, strip leading/trailing quotes, and convert all text to lower-case ASCII/Unicode standard.
- [ ] **2.5 Unit Testing on Tricky Entities**: Validate normalizer outputs on known edge cases: French accented characters, Indian landmark addresses, and hyphenated business names.

---

## Phase 3: High-Recall Multi-Key Inverted Index Blocking
**Objective**: Drastically reduce the $O(N_1 \cdot N_{2,3})$ comparison space to a small, high-quality candidate set per entity, maximizing the recall ceiling ($>95\%$).

- [ ] **3.1 Country-Partitioned Inverted Index**: Build discrete inverted indices separated by country (`US`, `IN`, `FR`) to completely eliminate cross-border false comparisons.
- [ ] **3.2 Multi-Key Generator**: Implement complementary, high-diversity blocking keys:
  - Exact Core Name Key.
  - Word 2-gram / 3-gram Name Keys (handles word order swaps).
  - First 4-character Name Prefix Key (handles trailing typos).
  - Address Token + First Name Token Anchors.
  - Postal Code / Numeric Token Anchors.
- [ ] **3.3 Super-Block Mitigation**: Filter out ultra-frequent generic stopwords (`store`, `services`, `enterprises`, `near`, `road`) that inflate posting list size and degrade efficiency.
- [ ] **3.4 Candidate Ranking & Top-K Pruning**: Score candidate pairs using fast token Jaccard similarity and retain only the top $K$ candidates (e.g., $K = 20\text{--}30$) per Source 1 entity.
- [ ] **3.5 Recall Benchmark on Ground Truth**: Execute blocking on a held-out training split; measure Candidate Recall ($Recall_{cand} = \frac{|Candidates \cap True|}{|True|}$) and Reduction Ratio ($RR$). Must achieve $\ge 95\%$ recall.

---

## Phase 4: Vectorized Pairwise Feature Engineering
**Objective**: Construct a discriminative numerical feature vector for each $(S_1, \text{Target})$ candidate pair to supply the machine learning classifier.

- [ ] **4.1 String Distance Features**: Extract normalized Levenshtein ratio, Jaro-Winkler similarity, and prefix matching scores for cleaned names and core names.
- [ ] **4.2 Token Set & N-gram Metrics**: Compute token Jaccard similarity, token sort ratio, token set ratio, and character 3-gram overlap between names.
- [ ] **4.3 Address Hierarchy Features**: Compute address token Jaccard, address character n-gram cosine, and check for street/unit/landmark token congruence.
- [ ] **4.4 Numeric & Postal Code Matching**: Extract and compare numbers (PIN codes, ZIP codes, building numbers); create exact-match boolean flags.
- [ ] **4.5 Structural Metadata Features**: Compute character length ratios, token count deltas, source origin indicator (`is_s2` vs `is_s3`), and empty address indicator flags.
- [ ] **4.6 Vectorized Batch Processing**: Benchmark feature extraction throughput; optimize with multiprocessing/C-extensions to achieve $>10,000$ pairs/second.

---

## Phase 5: GBDT Classifier Training & $F_{0.5}$ Threshold Optimization
**Objective**: Train an accurate pairwise classifier and tune decision thresholds specifically to maximize the competition's macro $F_{0.5}$ metric.

- [ ] **5.1 Grouped Train/Validation Split**: Partition training data by `source1_entity_id` (grouped split) to prevent pair leakage between train and validation folds.
- [ ] **5.2 Class Imbalance Handling**: Configure sample weights or focal loss in LightGBM/XGBoost to handle the extreme negative-to-positive ratio ($\sim 20:1$).
- [ ] **5.3 GBDT Model Training**: Train LightGBM with early stopping on validation loss using tuned tree depth, learning rate, and feature subsampling.
- [ ] **5.4 Probability Calibration**: Apply Isotonic Regression or Platt Scaling (`CalibratedClassifierCV`) to output well-calibrated posterior probabilities.
- [ ] **5.5 Macro $F_{0.5}$ Grid Search**: Sweep decision threshold $\theta \in [0.1, 0.95]$ at step $0.01$ to identify $\theta^*$ that maximizes the macro $F_{0.5}$ score including singletons.
- [ ] **5.6 Cross-Validation & Generalization Verification**: Verify consistency across 5 folds; confirm that performance remains robust across US and India subsets.

---

## Phase 6: Post-Processing, Test Inference & Submission Verification
**Objective**: Execute end-to-end inference on the test set, export compliant output TSV files, and verify format integrity.

- [x] **6.1 Test Set Blocking & Candidate Generation**: Run blocking across all test records, ensuring France records are seamlessly processed without exceptions.
- [x] **6.2 Pairwise Feature Extraction on Test Candidates**: Generate feature vectors for test candidate pairs in chunked memory-safe batches.
- [x] **6.3 Model Inference & Scoring**: Predict match probabilities for all test candidate pairs using calibrated ensemble models.
- [x] **6.4 Decision Thresholding & Singleton Prediction**: Apply optimal threshold $\theta^*$; output empty strings for entities whose candidate probabilities fail to exceed threshold.
- [x] **6.5 TSV Serialization**: Export `output/candidate_pairs.tsv` and `output/matching_results.tsv` strictly with tab separators (`\t`) and unquoted comma-separated lists.
- [x] **6.6 Format Validation with `validate_submission.py`**: Execute official validation script with `--matching`, `--candidate`, and `--test-dir`. Must achieve `PASS (exit 0)`.
- [x] **6.7 Submission Packaging**: Generate final `<team_name>_submission.zip` containing `output/`, `code/business_entity_resolution/`, and completed `Documentation_template.md`.
