# Library Documentation & Integration Rules: Amazon ML Challenge 2026

This document establishes integration standards and execution patterns for the core libraries utilized in the Business Entity Resolution pipeline.

---

## 1. LightGBM (`lightgbm >= 4.0.0`)
LightGBM is the primary gradient-boosted decision tree framework selected for pairwise binary classification due to its histogram-based splitting speed and low memory footprint.

### A. Recommended Hyperparameters for Imbalanced Entity Matching
Candidate generation yields a severe class imbalance ($\sim 15\text{--}30$ negatives per positive pair). Configure the classifier accordingly:

```python
import lightgbm as lgb

LGBM_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": 6,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "scale_pos_weight": 5.0,     # Compensate for candidate negative skew
    "random_state": 42,
    "n_jobs": 4,
    "verbose": -1,
}
```

### B. Training with Early Stopping & Model Serialization
```python
# Modern LightGBM Callback syntax (v4.0+)
callbacks = [
    lgb.early_stopping(stopping_rounds=30, verbose=False),
    lgb.log_evaluation(period=50)
]

clf = lgb.LGBMClassifier(**LGBM_PARAMS, n_estimators=500)
clf.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=callbacks
)

# Save booster to disk
import pickle
with open("output/lgbm_matcher.pkl", "wb") as f:
    pickle.dump(clf, f)
```

---

## 2. Scikit-Learn (`scikit-learn >= 1.3.0`)

### A. Grouped Partitioning (Zero Data Leakage)
Never perform standard random split across entity pairs. When an $S_1$ entity appears in both train and validation splits, the model memorizes entity patterns rather than learning pairwise similarity functions:

```python
from sklearn.model_selection import GroupKFold

gkf = GroupKFold(n_splits=5)
# groups must be the unique Source 1 entity IDs
for train_idx, val_idx in gkf.split(X, y, groups=df_pairs["source1_entity_id"]):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
```

### B. Probability Calibration
Raw tree model scores often skew toward 0 or 1. Use `CalibratedClassifierCV` (with Isotonic or Sigmoid method) to ensure predicted probabilities correspond to true match likelihoods:

```python
from sklearn.calibration import CalibratedClassifierCV

calibrated_clf = CalibratedClassifierCV(clf, cv="prefit", method="isotonic")
calibrated_clf.fit(X_val, y_val)
probs = calibrated_clf.predict_proba(X_test)[:, 1]
```

---

## 3. RapidFuzz & String Similarity Metrics
For high-throughput string distance calculations across millions of candidate pairs, prefer C-accelerated Levenshtein and Jaro-Winkler implementations:

```python
from rapidfuzz import fuzz, distance

# 1. Normalized Levenshtein similarity [0.0, 1.0]
lev_sim = fuzz.ratio(name1, name2) / 100.0

# 2. Token Sort Ratio (invariant to word order swaps)
token_sort = fuzz.token_sort_ratio(name1, name2) / 100.0

# 3. Token Set Ratio (handles subset names, e.g. "Google" vs "Google Cloud LLC")
token_set = fuzz.token_set_ratio(name1, name2) / 100.0

# 4. Jaro-Winkler similarity (rewards prefix matches)
jaro_winkler = distance.JaroWinkler.similarity(name1, name2)
```
*Fallback Note*: If `rapidfuzz` is not installed in the target environment, fallback to Python standard library `difflib.SequenceMatcher` or character n-gram Jaccard equivalents implemented in `src/matching_model.py`.

---

## 4. Pandas (`pandas >= 2.0.0`)

### A. Safe TSV Reading
Always enforce `sep="\t"` and disable default NaN conversions for empty strings:
```python
df = pd.read_csv(
    filepath,
    sep="\t",
    dtype=str,
    keep_default_na=False,   # Preserves empty string as "" instead of NaN
    na_values=[]
)
```

### B. Chunked Streaming for Inference
```python
chunk_size = 50000
for chunk in pd.read_csv(test_file, sep="\t", chunksize=chunk_size, dtype=str, keep_default_na=False):
    # Process batch candidates in memory
    process_chunk(chunk)
```

---

## 5. Submission Validator (`utils/validate_submission.py`)
Official validator provided in the challenge repository. It strictly audits submission formatting and checks cross-file constraints.

### A. Execution Syntax
Run from the project root:
```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

### B. Optional Flags
- `--check-ids`: Reads `test_source2.tsv` and `test_source3.tsv` to verify that every predicted target ID actually exists in the test set.
- `--quiet`: Suppresses informational warnings and only displays validation errors.

### C. Return Codes
- `Exit Code 0 (PASS)`: Files are completely valid and ready for submission to the competition portal.
- `Exit Code 1 (FAIL)`: Rule violations detected (e.g., missing $S_1$ entity IDs, delimiter error, duplicate IDs in a list, non-existent target IDs). The pipeline must be fixed before submitting.
