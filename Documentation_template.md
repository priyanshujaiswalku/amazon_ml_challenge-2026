# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [Your Team Name]  
**Team Members:** Priyanshu Kumar  
**Submission Date:** September 2026  

---

## 1. Executive Summary
We present an end-to-end, high-recall, precision-optimized solution for the Amazon ML Challenge 2026 Business Entity Resolution task. Our pipeline couples a country-partitioned, multi-key inverted index blocking engine (>99.999% comparison reduction, 90.16% top-15 candidate recall) with an 18-feature LightGBM pairwise classifier optimized specifically for the competition's macro $F_{0.5}$ metric, achieving **0.9573 Macro $F_{0.5}$** and **0.9757 Macro Precision** on held-out validation data while operating strictly within memory constraints (< 800 MB RAM).

---

## 2. Methodology

### 2.1 Problem Analysis
During exploratory data analysis (EDA), we uncovered key challenges in multi-source commercial data:
1. **Intractable Comparison Space**: 1.73M Source 1 records against ~10M Source 2 & 3 records yields $\approx 17.8 \text{ Trillion}$ pairwise comparisons, making brute force intractable.
2. **Noise Patterns**:
   - *Legal Suffix Inconsistencies*: Diverse legal entity suffixes (`LLC`, `Pvt Ltd`, `Corp`, `SARL`, `SASU`) across US, India, and France distort exact name matches.
   - *DBA & Severe Typos*: Certain records feature completely corrupted trade names or acronyms while maintaining identical physical street locations.
   - *Address Irregularities*: Road abbreviations (`St` vs `Street`, `Rd` vs `Road`), variable municipal numbering, unit/suite ordering variations, and missing PIN codes or full addresses.
3. **Country Separation**: Businesses operate within distinct regional domains (US, India, France); cross-country matching does not occur, enabling natural domain partitioning.

### 2.2 Solution Strategy
- **Approach Type**: Two-Stage Hybrid (Multi-Key Inverted Index Blocking + Pairwise Gradient Boosted Tree Classifier).
- **Core Innovation**:
  1. *Discriminative Address Anchoring*: Pairing building/door numbers with salient street tokens and postal codes to rescue matches where business names are corrupted or DBA.
  2. *Signed Geometric & Location Features*: Explicitly encoding number conflicts (e.g. `100 Market St` vs `500 Market St` $\rightarrow -1.0$) to aggressively penalize false merges.
  3. *Metric-Targeted Thresholding*: Calibrated grid-search optimization finding $\tau^* = 0.75$, weighting precision 2× over recall to maximize competition $F_{0.5}$.

---

## 3. Candidate Generation (Blocking)

To reduce the $17.8 \text{ Trillion}$ space down to ~11 candidates per entity without losing true matches, we developed a multi-key union strategy:

- **Blocking Keys Used**:
  - `Exact Core Name (CN)`: Standardized business name stripped of country-specific legal suffixes.
  - `First-2 Words (2W) & First Word (1W)`: Captures core brand identity and leading name tokens.
  - `Character Prefixes (PFX4, PFX5)`: Resilient to typographical noise and phonetic misspellings.
  - `Postal Code + Name Token (PIN_NAME)`: Hyper-local geographic anchor connecting PIN/ZIP codes with primary name tokens.
  - `Address Number + Street Word (ADDR, ADDR2W)`: Captures shared physical address anchors.
- **Candidate Pairs Generated**: Average of **10.9 candidates** per Source 1 entity (search space reduction $> 99.999\%$).
- **Recall Guarantee**: Validated on training ground truth, our blocking engine achieved **90.16% recall within the top-15 candidates** and **96.10% recall across candidate pools**, operating at **4,245 queries/sec** with zero memory overflow.

---

## 4. Matching Model

### 4.1 Features Used (18 Discriminative Pairwise Signals)
- **Name Features**:
  - `feat_name_exact_core` / `feat_name_exact_clean`: Exact match indicators post-normalization.
  - `feat_name_token_jaccard` & `feat_name_overlap_count`: Token overlap set metrics.
  - `feat_name_len_ratio`: Relative string length symmetry.
  - `feat_name_seq_ratio`: Ratcliff-Obershelp / SequenceMatcher edit distance ratio.
  - `feat_name_trigram_jaccard`: Character 3-gram overlap (captures minor typos and transliterations).
  - `feat_name_first_word_match` & `feat_name_prefix4_match`: Leading token identity flags.
- **Address Features**:
  - `feat_addr_token_jaccard` & `feat_addr_trigram_jaccard`: Address token and sub-word overlap.
  - `feat_addr_number_match`: Tri-state feature ($+1.0$ if street numbers match, $-1.0$ if both have numbers but they conflict, $0.0$ if missing).
  - `feat_addr_pincode_match`: Tri-state postal code concordance indicator.
  - `feat_addr_missing`: Explicit missingness indicator for unpopulated address fields.
- **Metadata & Blocking Features**:
  - `feat_blocking_score`: Multi-key agreement count.
  - `feat_is_source3`: Source indicator distinguishability flag.

### 4.2 Model Type & Training
- **Model**: LightGBM Classifier (`n_estimators=150`, `learning_rate=0.08`, `num_leaves=31`, `class_weight='balanced'`).
- **Entity-Level Group Splitting**: Training/validation splits are strictly partitioned by `source1_entity_id` to guarantee zero data leakage between train and test candidate pairs.
- **Threshold Selection Method**: Evaluated thresholds from $0.20$ to $0.90$ on holdout validation entities using exact macro-averaged $F_{0.5}$. The optimal operating point was determined at **$\tau^* = 0.75$**.

---

## 5. Results & Error Analysis

### 5.1 Validation Performance
- **Macro $F_{0.5}$ Score**: **0.9573** (95.73%)
- **Macro Precision**: **0.9757** (97.57%)
- **Macro Recall**: **0.9184** (91.84%)
- **Singleton Accuracy**: 100% on evaluated singletons (no false merges on entities without true targets).

### 5.2 Error Analysis
- **Common False Positives (Wrong Merges)**: Distinct businesses sharing a large corporate campus or business park without specific unit numbers, sharing high token overlap.
- **Common False Negatives (Missed Matches)**: Extremely sparse records where both business name has a severe DBA divergence and the address is completely unpopulated (`NaN`).

---

## 6. Conclusion
By pairing an intelligent, country-partitioned multi-key inverted index with a precision-tuned LightGBM pairwise classifier, our solution achieves competitive entity resolution accuracy (0.9573 Macro $F_{0.5}$) while remaining scalable, lightning-fast, and strictly within minimal hardware memory limits.

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
5. **Verify Submission Compliance**:
   ```bash
   python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv
   ```
