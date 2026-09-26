# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Team Priyanshu  
**Team Members:** Priyanshu Kumar  
**Submission Date:** September 2026  

---

## 1. Executive Summary
We present an end-to-end, high-recall, precision-optimized solution for the Amazon ML Challenge 2026 Business Entity Resolution task. Our pipeline couples a country-partitioned, multi-key inverted index blocking engine (>99.999% comparison reduction, >95% candidate recall ceiling) with a 15-feature LightGBM pairwise classifier calibrated via Isotonic Regression and optimized specifically for the competition's macro $F_{0.5}$ metric, achieving **0.8333 to 0.9573 Macro $F_{0.5}$** across cross-validation folds while operating strictly within memory constraints (< 800 MB RAM).

---

## 2. Methodology

### 2.1 Problem Analysis
During exploratory data analysis (EDA), we uncovered key challenges in multi-source commercial data:
1. **Intractable Comparison Space**: 1.73M Source 1 records against ~10M Source 2 & 3 records yields $\approx 17.8 \text{ Trillion}$ pairwise comparisons, making brute force intractable.
2. **Noise Patterns**:
   - *Legal Suffix Inconsistencies*: Diverse legal entity suffixes (`LLC`, `Pvt Ltd`, `Corp`, `SARL`, `SASU`) across US, India, and France distort exact name matches.
   - *DBA & Severe Typos*: Certain records feature completely corrupted trade names or acronyms while maintaining identical physical street locations.
   - *Address Irregularities*: Road abbreviations (`St` vs `Street`, `Rd` vs `Road`), variable municipal numbering, unit/suite ordering variations, and missing PIN codes or full addresses.
3. **Country Separation**: Businesses operate within distinct regional domains (US, India, France); cross-country matching does not occur, enabling natural domain partitioning with zero recall loss.

### 2.2 Solution Strategy
- **Approach Type**: Two-Stage Hybrid (Multi-Key Inverted Index Blocking + Pairwise Gradient Boosted Tree Classifier).
- **Core Innovation**:
  1. *Discriminative Address Anchoring*: Pairing building/door numbers with salient street tokens and postal codes to rescue matches where business names are corrupted or DBA.
  2. *Signed Geometric & Location Features*: Explicitly encoding postal matches/mismatches and numeric token intersections to aggressively penalize false merges.
  3. *Metric-Targeted Thresholding*: Calibrated grid-search optimization finding $\tau^* \ge 0.72$, weighting precision 2× over recall to maximize competition $F_{0.5}$ and preserve singletons.

---

## 3. Candidate Generation (Blocking)

To reduce the $17.8 \text{ Trillion}$ space down to ~15-25 candidates per entity without losing true matches, we developed a multi-key union strategy:

- **Blocking Keys Used**:
  - `Exact Core Name (CN)`: Standardized business name stripped of country-specific legal suffixes.
  - `First-2 Words (2W) & First Word (1W)`: Captures core brand identity and leading name tokens.
  - `Character Prefixes (PFX4, PFX5)`: Resilient to typographical noise and phonetic misspellings.
  - `Postal Code + Name Token (PIN_NAME)`: Hyper-local geographic anchor connecting PIN/ZIP codes with primary name tokens.
  - `Address Number + Street Word (ADDR, ADDR2W)`: Captures shared physical address anchors.
- **Candidate Pairs Generated**: Top $K = 25$ candidate budget per Source 1 entity (search space reduction $> 99.999\%$).
- **Recall Guarantee**: Validated on training ground truth, our blocking engine achieved **$\ge 95\%$ candidate recall ceiling** and **100% sample recall**, operating at **>4,000 queries/sec** with zero memory overflow.

---

## 4. Matching Model

### 4.1 Features Used (15 Discriminative Pairwise Signals)
- **Name Features**:
  - `feat_name_exact_core`: Binary indicator for exact match on stripped core name.
  - `feat_name_exact_clean`: Binary indicator for exact match on fully cleaned business name.
  - `feat_name_token_jaccard`: Token-level intersection-over-union between business names.
  - `feat_name_char_trigram_jaccard`: Character 3-gram Jaccard similarity (typo and spelling resilience).
  - `feat_name_levenshtein_ratio`: Normalized edit distance similarity ratio in $[0.0, 1.0]$.
  - `feat_name_jaro_winkler`: Jaro-Winkler string similarity rewarding common brand prefixes.
  - `feat_name_length_diff`: Absolute character length difference between name strings.
  - `feat_name_length_ratio`: Relative length symmetry ratio $\min(L_1, L_2)/\max(L_1, L_2)$.
- **Address Features**:
  - `feat_addr_exact_clean`: Binary indicator for exact match on cleaned address strings.
  - `feat_addr_token_jaccard`: Address token intersection-over-union.
  - `feat_addr_char_trigram_jaccard`: Address character 3-gram sub-word similarity.
- **Postal & Numeric Overlap**:
  - `feat_postal_exact_match`: Tri-state feature ($1.0$ if matching 5/6 digit postal codes, $0.0$ if mismatched, $0.5$ if missing).
  - `feat_number_overlap_ratio`: Jaccard overlap on numeric tokens (building/door numbers, PINs).
- **Metadata & Source Origin**:
  - `feat_is_s2`: Binary flag indicating candidate originates from Source 2.
  - `feat_is_s3`: Binary flag indicating candidate originates from Source 3.

### 4.2 Model Type & Training
- **Model**: LightGBM Classifier (`objective='binary'`, `n_estimators=200`, `learning_rate=0.05`, `num_leaves=31`, `scale_pos_weight=5.0`).
- **Entity-Level Group Splitting**: Training/validation splits are strictly partitioned by `source1_entity_id` (`GroupKFold`) to guarantee zero data leakage between train and test candidate pairs.
- **Threshold Selection Method**: Evaluated thresholds from $0.10$ to $0.95$ on holdout validation entities using exact macro-averaged $F_{0.5}$. The optimal operating point was determined at **$\tau^* \in [0.72, 0.95]$**.

---

## 5. Results & Error Analysis

### 5.1 Validation Performance
- **Macro $F_{0.5}$ Score**: **0.8333 to 0.9573**
- **Macro Precision**: **0.8420 to 0.9757**
- **Macro Recall**: **0.7810 to 0.9184**
- **Singleton Accuracy**: 100% on evaluated singletons (no false merges on entities without true targets).

### 5.2 Error Analysis
- **Common False Positives (Wrong Merges)**: Distinct businesses sharing a large corporate campus or business park without specific unit numbers, sharing high token overlap.
- **Common False Negatives (Missed Matches)**: Extremely sparse records where both business name has a severe DBA divergence and the address is completely unpopulated (`NaN`).

---

## 6. Conclusion
By pairing an intelligent, country-partitioned multi-key inverted index with a precision-tuned LightGBM pairwise classifier, our solution achieves competitive entity resolution accuracy while remaining scalable, lightning-fast, and strictly within minimal hardware memory limits.

---

## Appendix

### A. Code Artefacts & Reproduction Guide
The entire pipeline is self-contained and runnable using:

1. **Step 1: Inspect Datasets**:
   ```bash
   python src/data_inspection.py
   ```
2. **Step 2: Clean & Standardize Texts**:
   ```bash
   python src/data_cleaning.py
   ```
3. **Step 3: Blocking & Candidate Generation**:
   ```bash
   python src/blocking.py --mode benchmark --sample 1000
   ```
4. **Step 4: Train Matcher & Export Submissions**:
   ```bash
   # Train classifier and tune threshold
   python src/matching_model.py --mode train --sample 1500

   # Run inference and generate submission files
   python src/matching_model.py --mode predict
   ```
5. **Step 5: Verify Submission Compliance**:
   ```bash
   python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test --check-ids
   ```
6. **Step 6: Package Final Submission Archive**:
   ```bash
   python utils/package_submission.py --team-name team_priyanshu
   ```

