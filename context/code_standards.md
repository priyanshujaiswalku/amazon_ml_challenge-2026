# Code Standards & Engineering Conventions: Amazon ML Challenge 2026

## 1. Language & Typing Standards
- **Python Version**: Strict target of Python 3.10 or 3.11.
- **Type Annotations**: Every function and method signature must declare explicit parameter types and return type annotations:
  ```python
  from typing import List, Dict, Set, Tuple, Optional, Generator
  import pandas as pd

  def extract_features(
      df_s1: pd.DataFrame,
      candidates: Dict[str, List[str]],
      batch_size: int = 10000
  ) -> pd.DataFrame:
      ...
  ```
- **Docstrings**: Adhere to Google/NumPy docstring format for all public functions, detailing `Args:`, `Returns:`, and `Raises:`.

---

## 2. Naming Conventions
- **Modules & Scripts**: `snake_case.py` (e.g., `data_cleaning.py`, `matching_model.py`).
- **Classes**: `PascalCase` (e.g., `BlockingEngine`, `PairwiseFeatureExtractor`).
- **Functions & Methods**: `snake_case()` (e.g., `clean_business_name()`, `optimize_threshold()`).
- **Global Constants**: `SCREAMING_SNAKE_CASE` (e.g., `LEGAL_SUFFIXES`, `MAX_CANDIDATES_PER_ENTITY`, `RANDOM_SEED`).
- **Feature Column Names**: Prefix all engineered feature columns with `feat_` (e.g., `feat_name_jaro_winkler`, `feat_addr_jaccard`, `feat_postal_exact_match`).

---

## 3. High-Performance Vectorization & Pandas Rules
Entity resolution over millions of records requires strict performance discipline:

1. **Avoid Inefficient Iteration**:
   - ❌ **Forbidden in Critical Paths**: `df.iterrows()`, `df.itertuples()`, or `df.apply(lambda ...)` when vectorized equivalents exist.
   - ✅ **Preferred**: Vectorized Series operations (`df['col'].str.lower()`, `np.where(...)`), NumPy array operations, or batch comprehensions using native zip tuples:
     ```python
     # Fast native iteration over aligned arrays
     pairs = list(zip(df_s1['name'].values, df_targets['name'].values))
     scores = [fast_similarity(n1, n2) for n1, n2 in pairs]
     ```
2. **Pre-Compiled Regular Expressions**:
   All regular expressions used in string cleaning must be pre-compiled at module scope (`re.compile(...)`) to avoid regex compilation overhead inside loops.
3. **Memory-Conscious Data Types**:
   - Downcast integer identifiers and boolean flags where possible (`np.float32`, `np.int32`, `bool`).
   - Use categorical dtype for fixed-cardinality fields such as `country`.

---

## 4. Tab-Separated File I/O & Delimiter Defense
TSV formatting errors cause instant submission rejection. Follow these invariants:

1. **Explicit Tab Delimiter**:
   - Always read files with `sep="\t"`:
     ```python
     df = pd.read_csv(filepath, sep="\t", dtype=str, keep_default_na=False)
     ```
   - Always write files with `sep="\t"` and disable index:
     ```python
     df.to_csv(filepath, sep="\t", index=False)
     ```
2. **Quoting Invariant**:
   - Do **not** allow Pandas to wrap text in double quotes unless strictly necessary. For ID lists, use plain comma-separated strings without quotes:
     ```python
     # Correct:
     # S1-00001\tS2-00047,S3-00812
     # Incorrect:
     # "S1-00001"\t"S2-00047,S3-00812"
     ```
3. **Empty String for Singletons**:
   - Do **not** write `NaN`, `null`, `None`, or whitespace for singletons. The `matched_entity_ids` column for a singleton must be an empty string `""` immediately preceded by a tab:
     ```python
     # Representation in TSV:
     # S1-00003\t\n
     ```

---

## 5. Determinism & Reproducibility
Random variation between training runs compromises threshold optimization:

- **Global Random Seed**: Pinned to `42` across all stochastic processes.
  ```python
  import random
  import numpy as np

  RANDOM_SEED = 42
  random.seed(RANDOM_SEED)
  np.random.seed(RANDOM_SEED)
  ```
- **LightGBM / XGBoost Determinism**:
  Always pass `random_state=42`, `n_jobs=4`, and fixed tree hyperparameters.
- **Deterministic Grouped Splitting**:
  Cross-validation splits must use `GroupKFold` or `train_test_split` on unique `source1_entity_id` values, ensuring zero leakage of entities across folds.

---

## 6. Memory Safety & Large Dataset Processing
- **Streaming for Test Inference**:
  When processing the $\sim 1.73$M test records, use streaming chunks (`chunksize=100000`) rather than loading the entire feature matrix into memory at once.
- **Explicit Garbage Collection**:
  Delete unneeded large objects and invoke `gc.collect()` before initiating heavy matrix operations:
  ```python
  import gc
  del raw_df, candidates_dict
  gc.collect()
  ```

---

## 7. Command-Line Interface (CLI) Standards
Every executable script inside `src/` must provide a user-friendly CLI powered by `argparse`:
- Must support `--help` detailing available arguments and defaults.
- Must provide sensible defaults so the script can execute out-of-the-box from the project root.
- Standard arguments:
  - `--mode` (e.g., `train`, `predict`, `benchmark`, `inspect`).
  - `--data-dir` (path to raw data directory, default: `dataset/`).
  - `--output-dir` (path to export directory, default: `output/`).
  - `--threshold` (decision threshold for classification, default: tuned $\theta^*$).

---

## 8. Defensive Validation & Error Handling
- Never silence errors with bare `except: pass`. Log or print the exception details.
- Validate dataset existence and path validity before running lengthy training jobs.
- Before writing final submission files, assert:
  1. `len(output_df) == len(test_source1_df)` (Every $S_1$ entity exists).
  2. `output_df['source1_entity_id'].nunique() == len(output_df)` (Zero duplicate rows).
  3. No self-matches (`S1-` IDs present in `matched_entity_ids`).
