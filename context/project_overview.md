# Project Overview: Amazon ML Challenge 2026 — Business Entity Resolution

## 1. Executive Summary
The **Amazon ML Challenge 2026 — Business Entity Resolution** project is an end-to-end, high-performance, offline machine learning pipeline engineered to resolve duplicate and fragmented business identities across disparate commercial data sources. In commercial platforms, business records arrive from multiple independent sources with conflicting, missing, or corrupted information, lacking shared unique identifiers.

The core objective is to match records from a deduplicated reference source (**Source 1**) to corresponding entity records in two noisy secondary sources (**Source 2** and **Source 3**). A single Source 1 record may correspond to zero matches (singleton), one match, or multiple matches across Source 2 and Source 3.

---

## 2. Core Problem Definition
- **Reference Standard**: Source 1 ($S_1$) is the ground-truth anchor for evaluation. Matches must be identified for every Source 1 entity present in the test set.
- **Target Sources**: Source 2 ($S_2$) and Source 3 ($S_3$). Matches can originate from $S_2$, $S_3$, or both. Matches can never link back to $S_1$ or cross-link between $S_2$ and $S_3$.
- **Data Modality**: Tab-separated tabular datasets containing:
  - `entity_id`: Alphanumeric unique identifier prefixed with source origin (`S1-`, `S2-`, `S3-`).
  - `business_name`: Free-text commercial name with heavy noise (abbreviations, legal forms, typos, transliterations, word order variations).
  - `business_address`: Free-text geographic address with structural noise (landmarks, missing postal codes, unstandardized street types, abbreviations).
  - `country`: Categorical country identifier.
- **Geographic Scope & Zero-Shot Transfer**:
  - Training dataset covers **US** and **India**.
  - Test set introduces **France** (an open-set, out-of-distribution country label not seen in training). The pipeline must generalize zero-shot to France without hardcoded filters or localized assumptions that break on international entities.
- **Cardinality & Relationships**:
  - $1 \to 0$ (Singletons: entities with no match in $S_2$ or $S_3$).
  - $1 \to 1$ (One-to-one match in either $S_2$ or $S_3$).
  - $1 \to M$ (One-to-many matches across $S_2$, $S_3$, or multiple records within $S_2$/$S_3$).

---

## 3. Primary Evaluation Metric: Macro $F_{0.5}$ Score
Submissions are evaluated using the **Macro-Averaged $F_{\beta}$ Score ($\beta = 0.5$)** computed over all Source 1 entities in the evaluation set:

$$F_{0.5} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$

### Strategic Metric Implications:
1. **Precision-Weighted ($2\times$ importance over Recall)**:
   In real-world entity resolution, merging distinct commercial businesses (false positive) creates severe data corruption and liability, far worse than failing to link two records (false negative). $F_{0.5}$ penalizes false matches aggressively.
2. **Macro-Averaging over All $S_1$ Entities**:
   $F_{0.5}$ is calculated individually per $S_1$ entity and averaged across all entities. Every entity contributes equally to the leaderboard score.
3. **Singleton Treatment ($1.0$ vs $0.0$)**:
   - If an $S_1$ entity has no true matches in ground truth, predicting an empty list yields **$F_{0.5} = 1.0$**.
   - If any candidate is falsely matched to a singleton entity, precision becomes $0.0$, driving that entity's score to **$0.0$**.
   - Preserving true singletons and maintaining conservative prediction thresholds is a primary ranking differentiator.

---

## 4. Primary Pipeline Execution Flow
The end-to-end resolution pipeline follows a strict five-stage linear flow:

```
[Raw TSV Datasets (S1, S2, S3)]
              │
              ▼
    1. Data Health & Audit (data_inspection.py)
              │
              ▼
    2. Vectorized Normalization (data_cleaning.py)
       - Legal suffix canonicalization (US, IN, FR)
       - Address token standardization & landmark extraction
       - Open-set country preservation
              │
              ▼
    3. High-Recall Multi-Key Inverted Index Blocking (blocking.py)
       - Country partitioning (strict zero cross-country bleed)
       - Exact core name, token n-grams, address anchors
       - Candidate ranking & top-K pruning (target >95% recall ceiling)
              │
              ▼
    4. Pairwise Feature Engineering & Classification (matching_model.py)
       - String distance, token sets, numeric/postal overlap, length ratios
       - Gradient Boosted Decision Tree (LightGBM / XGBoost)
       - Probability calibration (Isotonic / Platt)
       - Precision-tuned threshold optimization for max macro F_0.5
              │
              ▼
    5. Submission Post-Processing & Validation (validate_submission.py)
       - candidate_pairs.tsv (blocking candidates)
       - matching_results.tsv (final filtered matches)
       - Full compliance pass: single tab delimiter, no quotes, zero missing S1
```

---

## 5. Project Scope Boundaries

### In Scope (Mandatory Competition Deliverables)
- **High-Speed Vectorized Preprocessing**: Modular regex and string manipulation optimized for millions of text records.
- **Scalable Multi-Key Inverted Index Blocking**: Sub-linear candidate filtering reducing $\sim 17.3$ trillion potential pairs down to $15\text{--}30$ candidate pairs per entity.
- **Pairwise Feature Engineering**: Fast extraction of lexical, token-based, character n-gram, phonetic, and address-specific similarity features.
- **Supervised ML Classification**: LightGBM, XGBoost, and CatBoost binary classifiers with calibrated probabilities.
- **$F_{0.5}$ Threshold Optimizer**: Automated search algorithm to locate optimal decision boundary $\theta^*$ maximizing macro $F_{0.5}$ on cross-validation folds.
- **Strict Format Compliance & Local Validation**: Pre-submission automated testing via `utils/validate_submission.py`.
- **Reproducible Package Export**: Production of `<team_name>_submission.zip` matching official submission package structure.

### Strictly Out of Scope & Prohibited
- **Zero External Lookups**: STRICT PROHIBITION against external commercial APIs, Google Maps, OpenStreetMap, geocoding lookups, Wikipedia, government corporate registers (e.g., MCA India, US SEC EDGAR, INSEE France), or web scraping. Violation triggers immediate disqualification.
- **No Hardcoded Closed-Set Geographies**: The pipeline must never filter or one-hot encode solely for `{US, India}`. France and any arbitrary country code must execute through the identical generalizable pipeline.
- **No Heavy / Non-Compliant Deep Learning Models**: Models exceeding 8 Billion parameters or lacking permissive open-source licenses (MIT or Apache 2.0) are disallowed.
- **No Web UI / Frontend Overhead**: This is a competitive machine learning and data engineering pipeline. No user interface, React, Vue, CSS, or web server components are required or permitted in the code package.
