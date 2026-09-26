# Feature & Modeling Rules: Amazon ML Challenge 2026

This document defines the strict mathematical, algorithmic, and machine learning rules governing feature generation, candidate blocking, classifier training, and decision thresholding.

---

## 1. Golden Rules of Business Entity Resolution

### Rule 1: Blocking Recall is the Hard Ceiling of Pipeline Performance
- A true match not retrieved during blocking can **never** be recovered by the classifier.
- Target: The candidate blocking stage must achieve **$\ge 95\%$ Candidate Recall** on ground-truth training evaluations.
- Metric formula:
  $$\text{Candidate Recall} = \frac{|\mathcal{C} \cap \mathcal{M}^*|}{|\mathcal{M}^*|}$$
  where $\mathcal{C}$ is the candidate set and $\mathcal{M}^*$ is the ground-truth match set.

### Rule 2: Country Isolation Invariant
- Real-world businesses do not match across international borders.
- Records must strictly be partitioned by country:
  $$\text{Candidate}(e_1, e_{\text{target}}) = \emptyset \quad \forall e_1, e_{\text{target}} \text{ where } \text{Country}(e_1) \neq \text{Country}(e_{\text{target}})$$
- This reduces search space by $\sim 67\%$ with zero loss in recall.

### Rule 3: Source Reference Invariant
- The challenge requires matching Source 1 entities to Source 2 and Source 3.
- Predictions must **only** contain `S2-` and `S3-` entity IDs.
- Any occurrence of an `S1-` ID in `matched_entity_ids` triggers immediate rejection by the validator.

### Rule 4: Precision-Heavy $F_{0.5}$ Optimization
- The evaluation metric weights Precision twice as heavily as Recall ($\beta = 0.5$):
  $$F_{0.5} = \frac{(1 + 0.5^2) \cdot P \cdot R}{0.5^2 \cdot P + R} = \frac{1.25 \cdot P \cdot R}{0.25 \cdot P + R}$$
- **Operational Rule**: A false positive (merging non-identical businesses) is mathematically penalized $4\times$ more than a false negative. The classifier decision threshold $\theta^*$ must be tuned conservatively (typically $\theta^* \in [0.65, 0.85]$ rather than the standard $0.5$).

### Rule 5: Singleton Integrity
- If an $S_1$ entity has no true matches, predicting an empty string yields $F_{0.5} = 1.0$.
- Predicting a single false match drops that entity's score from $1.0 \to 0.0$.
- **Operational Rule**: When all candidate match probabilities for an $S_1$ entity fall below the optimal threshold $\theta^*$, the pipeline must emit an empty match list `""`.

### Rule 6: Candidate Set Superset Guarantee
- Every entity ID appearing in `output/matching_results.tsv` for an $S_1$ entity **must** be present in `output/candidate_pairs.tsv` for that same $S_1$ entity:
  $$\text{PredictedMatches}(e_1) \subseteq \text{Candidates}(e_1)$$
- An ID appearing in final predictions without being a candidate indicates an architectural violation.

### Rule 7: Absolute Prohibition of External Data
- External APIs, Google Maps, OpenStreetMap, geocoders, and government registry lookups are strictly prohibited. All features must be derived purely from the provided text columns.

---

## 2. Blocking Key Generation Rules
To balance candidate recall and execution speed, the multi-key inverted index combines:

1. **Exact Clean Core Name**: Stripped of legal suffixes, corporate stop-words, and punctuation.
2. **First Word Anchor**: First non-stopword token in cleaned business name.
3. **Word 2-Gram Anchors**: Consecutive word pairs in business names of length $\ge 3$ characters (resilient to word-order flips).
4. **Name Prefix (4 chars)**: First 4 alphanumeric characters of the core name (resilient to suffix typos).
5. **Postal / PIN Code Key**: Exact match on 5-digit ZIP (US/FR) or 6-digit PIN (IN) combined with the first name token.
6. **Candidate Budget ($K$)**:
   - Postings lists for generic stop-words (e.g., `store`, `restaurant`, `services`) must be suppressed.
   - For each $S_1$ entity, candidate pairs must be scored by token Jaccard and capped at top $K$ ($K \in [15, 30]$).

---

## 3. Pairwise Feature Engineering Standards
For each candidate pair $(S_1, \text{Target})$, the pipeline extracts a 15-dimensional numeric feature vector:

| Feature Name | Type | Value Range | Formulation & Rationale |
|---|---|---|---|
| `feat_name_exact_core` | Boolean (0/1) | $\{0, 1\}$ | Exact match on stripped core name. |
| `feat_name_exact_clean` | Boolean (0/1) | $\{0, 1\}$ | Exact match on fully cleaned name. |
| `feat_name_token_jaccard` | Float | $[0.0, 1.0]$ | Token intersection over union: $\frac{\|T_1 \cap T_2\|}{\|T_1 \cup T_2\|}$. |
| `feat_name_char_trigram_jaccard` | Float | $[0.0, 1.0]$ | Character 3-gram Jaccard (high typo resilience). |
| `feat_name_levenshtein_ratio` | Float | $[0.0, 1.0]$ | Normalized edit distance similarity. |
| `feat_name_jaro_winkler` | Float | $[0.0, 1.0]$ | String similarity with prefix bonus. |
| `feat_name_length_diff` | Integer | $[0, \infty)$ | Absolute difference in name string character lengths. |
| `feat_name_length_ratio` | Float | $[0.0, 1.0]$ | $\frac{\min(L_1, L_2)}{\max(L_1, L_2)}$. |
| `feat_addr_exact_clean` | Boolean (0/1) | $\{0, 1\}$ | Exact match on cleaned address strings. |
| `feat_addr_token_jaccard` | Float | $[0.0, 1.0]$ | Address token intersection over union. |
| `feat_addr_char_trigram_jaccard` | Float | $[0.0, 1.0]$ | Character 3-gram Jaccard on address text. |
| `feat_postal_exact_match` | Float | $\{0.0, 0.5, 1.0\}$ | $1.0$ if both postal codes match; $0.0$ if mismatched; $0.5$ if missing. |
| `feat_number_overlap_ratio` | Float | $[0.0, 1.0]$ | Intersection ratio of all numeric tokens (building numbers, PINs). |
| `feat_is_s2` | Boolean (0/1) | $\{0, 1\}$ | $1$ if candidate originates from Source 2, else $0$. |
| `feat_is_s3` | Boolean (0/1) | $\{0, 1\}$ | $1$ if candidate originates from Source 3, else $0$. |

---

## 4. Modeling & Threshold Optimization Protocol

1. **Grouped Cross-Validation**:
   Use 5-fold `GroupKFold` split grouped on `source1_entity_id`. Zero pair leakage permitted.
2. **Classifier Training**:
   Train LightGBM with `objective='binary'`, `scale_pos_weight=5.0`, and tree regularization.
3. **Probability Calibration**:
   Apply Isotonic calibration to obtain faithful posterior match probabilities $\hat{p} \in [0.0, 1.0]$.
4. **$F_{0.5}$ Metric Threshold Search**:
   Iterate $\theta \in [0.10, 0.95]$ with step $0.01$. Compute macro-averaged $F_{0.5}$ across all validation $S_1$ entities (including singletons) at each step. Select $\theta^*$ achieving maximum validation score:
   $$\theta^* = \arg\max_{\theta} \text{Macro-}F_{0.5}(\theta)$$
