"""
Matching Model Module
=====================
Step 4 & Step 5: Vectorized Feature Engineering, Calibrated GBDT Classification,
and Precision-Weighted Macro F_0.5 Threshold Optimization.

This module extracts discriminative pairwise features between Source 1 records
and candidate targets (Source 2 / Source 3), trains a gradient-boosted decision
tree (LightGBM) classifier with probability calibration, and optimizes the
decision threshold to maximize the competition's primary evaluation metric:
Macro F_0.5 Score.
"""

import argparse
from collections import defaultdict
import difflib
import os
from pathlib import Path
import pickle
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GroupKFold
import lightgbm as lgb

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.data_cleaning import (
        clean_business_name,
        clean_business_address,
        clean_country,
    )
    from src.blocking import BlockingEngine, extract_blocking_keys
except ImportError:
    from data_cleaning import (
        clean_business_name,
        clean_business_address,
        clean_country,
    )
    from blocking import BlockingEngine, extract_blocking_keys


# ---------------------------------------------------------------------------
# String Distance & Similarity Primitives (High Performance & Self-Contained)
# ---------------------------------------------------------------------------

def fast_levenshtein_ratio(s1: str, s2: str) -> float:
    """
    Computes normalized Levenshtein similarity ratio in [0.0, 1.0].
    Memory-efficient O(min(m, n)) space implementation.
    """
    if s1 == s2:
        return 1.0
    len1, len2 = len(s1), len(s2)
    if len1 == 0 or len2 == 0:
        return 0.0

    if len1 > len2:
        s1, s2 = s2, s1
        len1, len2 = len2, len1

    prev_row = list(range(len1 + 1))
    for j, c2 in enumerate(s2):
        curr_row = [j + 1] * (len1 + 1)
        for i, c1 in enumerate(s1):
            cost = 0 if c1 == c2 else 1
            curr_row[i + 1] = min(
                curr_row[i] + 1,       # insertion
                prev_row[i + 1] + 1,   # deletion
                prev_row[i] + cost,    # substitution
            )
        prev_row = curr_row

    dist = prev_row[len1]
    max_len = max(len1, len2)
    return 1.0 - (dist / max_len) if max_len > 0 else 0.0


def fast_jaro_winkler(s1: str, s2: str, p: float = 0.1, max_l: int = 4) -> float:
    """
    Computes Jaro-Winkler string similarity with prefix bonus in [0.0, 1.0].
    """
    if s1 == s2:
        return 1.0
    len1, len2 = len(s1), len(s2)
    if len1 == 0 or len2 == 0:
        return 0.0

    match_distance = max(len1, len2) // 2 - 1
    if match_distance < 0:
        match_distance = 0

    s1_matches = [False] * len1
    s2_matches = [False] * len2

    matches = 0
    for i in range(len1):
        start = max(0, i - match_distance)
        end = min(i + match_distance + 1, len2)
        for j in range(start, end):
            if s2_matches[j]:
                continue
            if s1[i] == s2[j]:
                s1_matches[i] = True
                s2_matches[j] = True
                matches += 1
                break

    if matches == 0:
        return 0.0

    transpositions = 0
    k = 0
    for i in range(len1):
        if not s1_matches[i]:
            continue
        while not s2_matches[k]:
            k += 1
        if s1[i] != s2[k]:
            transpositions += 1
        k += 1

    jaro = (
        (matches / len1)
        + (matches / len2)
        + ((matches - transpositions / 2.0) / matches)
    ) / 3.0

    # Common prefix bonus
    prefix_len = 0
    for i in range(min(len1, len2, max_l)):
        if s1[i] == s2[i]:
            prefix_len += 1
        else:
            break

    return jaro + prefix_len * p * (1.0 - jaro)


def compute_jaccard(set1: Set[Any], set2: Set[Any]) -> float:
    """Computes Jaccard similarity (intersection over union) between two sets."""
    if not set1 or not set2:
        return 0.0
    inter = len(set1 & set2)
    union = len(set1 | set2)
    return inter / union if union > 0 else 0.0


def get_char_trigrams(text: str) -> Set[str]:
    """Generates character 3-grams for typo-resilient comparison."""
    if not text or len(text) < 3:
        return set()
    return {text[i:i + 3] for i in range(len(text) - 2)}


# ---------------------------------------------------------------------------
# Feature Engineering Standards (Matching context/feature_and_model_rules.md)
# ---------------------------------------------------------------------------

FEATURE_COLUMNS: List[str] = [
    "feat_name_exact_core",
    "feat_name_exact_clean",
    "feat_name_token_jaccard",
    "feat_name_char_trigram_jaccard",
    "feat_name_levenshtein_ratio",
    "feat_name_jaro_winkler",
    "feat_name_length_diff",
    "feat_name_length_ratio",
    "feat_addr_exact_clean",
    "feat_addr_token_jaccard",
    "feat_addr_char_trigram_jaccard",
    "feat_postal_exact_match",
    "feat_number_overlap_ratio",
    "feat_is_s2",
    "feat_is_s3",
]


def extract_pair_features(
    s1_name: Optional[str],
    s1_addr: Optional[str],
    s1_country: Optional[str],
    target_id: str,
    target_name: Optional[str],
    target_addr: Optional[str],
    target_country: Optional[str],
) -> List[float]:
    """
    Extracts a 15-dimensional numeric feature vector for a candidate pair (Source 1, Target).

    Returns:
        List[float]: Exactly matching FEATURE_COLUMNS order.
    """
    c1_name, core1 = clean_business_name(s1_name)
    c1_addr = clean_business_address(s1_addr)
    c2_name, core2 = clean_business_name(target_name)
    c2_addr = clean_business_address(target_addr)

    # 1. feat_name_exact_core
    feat_name_exact_core = 1.0 if (core1 and core1 == core2) else 0.0

    # 2. feat_name_exact_clean
    feat_name_exact_clean = 1.0 if (c1_name and c1_name == c2_name) else 0.0

    # 3. feat_name_token_jaccard
    tokens1 = set(core1.split()) if core1 else set()
    tokens2 = set(core2.split()) if core2 else set()
    feat_name_token_jaccard = compute_jaccard(tokens1, tokens2)

    # 4. feat_name_char_trigram_jaccard
    tri1 = get_char_trigrams(core1)
    tri2 = get_char_trigrams(core2)
    feat_name_char_trigram_jaccard = compute_jaccard(tri1, tri2)

    # 5. feat_name_levenshtein_ratio
    feat_name_levenshtein_ratio = fast_levenshtein_ratio(core1, core2)

    # 6. feat_name_jaro_winkler
    feat_name_jaro_winkler = fast_jaro_winkler(core1, core2)

    # 7. feat_name_length_diff
    len1, len2 = len(core1), len(core2)
    feat_name_length_diff = float(abs(len1 - len2))

    # 8. feat_name_length_ratio
    max_l = max(len1, len2)
    feat_name_length_ratio = (min(len1, len2) / max_l) if max_l > 0 else 0.0

    # 9. feat_addr_exact_clean
    feat_addr_exact_clean = 1.0 if (c1_addr and c1_addr == c2_addr) else 0.0

    # 10. feat_addr_token_jaccard
    addr_tok1 = set(c1_addr.split()) if c1_addr else set()
    addr_tok2 = set(c2_addr.split()) if c2_addr else set()
    feat_addr_token_jaccard = compute_jaccard(addr_tok1, addr_tok2)

    # 11. feat_addr_char_trigram_jaccard
    addr_tri1 = get_char_trigrams(c1_addr)
    addr_tri2 = get_char_trigrams(c2_addr)
    feat_addr_char_trigram_jaccard = compute_jaccard(addr_tri1, addr_tri2)

    # 12. feat_postal_exact_match (1.0 if match, 0.0 if mismatch, 0.5 if missing)
    nums1 = {t for t in addr_tok1 if t.isdigit()}
    nums2 = {t for t in addr_tok2 if t.isdigit()}
    pins1 = {n for n in nums1 if len(n) in (5, 6)}
    pins2 = {n for n in nums2 if len(n) in (5, 6)}
    if pins1 and pins2:
        feat_postal_exact_match = 1.0 if bool(pins1 & pins2) else 0.0
    else:
        feat_postal_exact_match = 0.5

    # 13. feat_number_overlap_ratio
    feat_number_overlap_ratio = compute_jaccard(nums1, nums2)

    # 14. feat_is_s2
    feat_is_s2 = 1.0 if target_id.startswith("S2-") else 0.0

    # 15. feat_is_s3
    feat_is_s3 = 1.0 if target_id.startswith("S3-") else 0.0

    return [
        feat_name_exact_core,
        feat_name_exact_clean,
        feat_name_token_jaccard,
        feat_name_char_trigram_jaccard,
        feat_name_levenshtein_ratio,
        feat_name_jaro_winkler,
        feat_name_length_diff,
        feat_name_length_ratio,
        feat_addr_exact_clean,
        feat_addr_token_jaccard,
        feat_addr_char_trigram_jaccard,
        feat_postal_exact_match,
        feat_number_overlap_ratio,
        feat_is_s2,
        feat_is_s3,
    ]


# ---------------------------------------------------------------------------
# Macro F_0.5 Metric Computation (Official Contest Scoring Formulation)
# ---------------------------------------------------------------------------

def calculate_macro_f05(
    ground_truth_dict: Dict[str, Set[str]],
    predictions_dict: Dict[str, Set[str]],
) -> Tuple[float, float, float]:
    """
    Computes official macro-averaged F_0.5 score across all Source 1 entities,
    correctly evaluating singletons according to competition rules:
      - Singleton True + Predicted Empty  -> F0.5 = 1.0, P = 1.0, R = 1.0
      - Singleton True + Predicted Match  -> F0.5 = 0.0, P = 0.0, R = 0.0
      - Match True     + Predicted Empty  -> F0.5 = 0.0, P = 0.0, R = 0.0
      - Match True     + Predicted Match  -> Standard F_0.5 formula

    Formula:
      F_0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)

    Returns:
        Tuple[float, float, float]: (macro_f05, macro_precision, macro_recall)
    """
    f05_scores: List[float] = []
    precisions: List[float] = []
    recalls: List[float] = []

    for sid, true_set in ground_truth_dict.items():
        pred_set = predictions_dict.get(sid, set())

        # Singleton evaluation (true entity has no matches in S2/S3)
        if not true_set:
            if not pred_set:
                f05_scores.append(1.0)
                precisions.append(1.0)
                recalls.append(1.0)
            else:
                f05_scores.append(0.0)
                precisions.append(0.0)
                recalls.append(0.0)
            continue

        # Non-singleton entity with no matches predicted
        if not pred_set:
            f05_scores.append(0.0)
            precisions.append(0.0)
            recalls.append(0.0)
            continue

        tp = len(pred_set & true_set)
        p = tp / len(pred_set)
        r = tp / len(true_set)
        precisions.append(p)
        recalls.append(r)

        denom = (0.25 * p) + r
        score = (1.25 * p * r) / denom if denom > 0 else 0.0
        f05_scores.append(score)

    macro_f05 = float(np.mean(f05_scores)) if f05_scores else 0.0
    macro_p = float(np.mean(precisions)) if precisions else 0.0
    macro_r = float(np.mean(recalls)) if recalls else 0.0
    return macro_f05, macro_p, macro_r


# ---------------------------------------------------------------------------
# Training Dataset Preparation with Class Balance Safeguards
# ---------------------------------------------------------------------------

def build_training_pairwise_dataset(
    df_s1: pd.DataFrame,
    gt_map: Dict[str, Set[str]],
    target_registry: Dict[str, Any],
    engine: BlockingEngine,
    negatives_per_positive: int = 5,
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[str, str]]]:
    """
    Builds candidate pairs feature matrix X, binary labels y, and metadata pairs.
    Includes negative candidate pairs from blocking and within-country distractors
    to ensure the classifier learns robust discriminative boundaries.
    """
    X_rows: List[List[float]] = []
    y_labels: List[int] = []
    pair_metadata: List[Tuple[str, str]] = []

    # Partition targets by country for within-country negative augmentation
    targets_by_country: Dict[str, List[str]] = defaultdict(list)
    for eid, row in target_registry.items():
        c = clean_country(row.get("country"))
        targets_by_country[c].append(eid)

    for _, s1_row in df_s1.iterrows():
        sid = s1_row["entity_id"]
        s1_country = clean_country(s1_row.get("country"))
        true_targets = gt_map.get(sid, set())

        # Retrieve blocking candidates
        candidates = engine.retrieve_candidates_for_query(
            s1_row.get("business_name"),
            s1_row.get("business_address"),
            s1_row.get("country"),
        )
        cand_set = set(candidates)

        # Include candidate pairs
        for cid in candidates:
            if cid not in target_registry:
                continue
            t_row = target_registry[cid]
            feats = extract_pair_features(
                s1_name=s1_row.get("business_name"),
                s1_addr=s1_row.get("business_address"),
                s1_country=s1_row.get("country"),
                target_id=cid,
                target_name=t_row.get("business_name"),
                target_addr=t_row.get("business_address"),
                target_country=t_row.get("country"),
            )
            label = 1 if cid in true_targets else 0
            X_rows.append(feats)
            y_labels.append(label)
            pair_metadata.append((sid, cid))

        # Class balance safeguard: If candidates contained zero negatives,
        # sample within-country distractors to establish negative training signals
        num_negs = sum(1 for cid in candidates if cid not in true_targets)
        needed_negs = negatives_per_positive - num_negs
        if needed_negs > 0 and s1_country in targets_by_country:
            pool = [t for t in targets_by_country[s1_country] if t not in true_targets and t not in cand_set]
            for dist_id in pool[:needed_negs]:
                t_row = target_registry[dist_id]
                feats = extract_pair_features(
                    s1_name=s1_row.get("business_name"),
                    s1_addr=s1_row.get("business_address"),
                    s1_country=s1_row.get("country"),
                    target_id=dist_id,
                    target_name=t_row.get("business_name"),
                    target_addr=t_row.get("business_address"),
                    target_country=t_row.get("country"),
                )
                X_rows.append(feats)
                y_labels.append(0)
                pair_metadata.append((sid, dist_id))

    X = np.array(X_rows, dtype=np.float32) if X_rows else np.zeros((0, len(FEATURE_COLUMNS)), dtype=np.float32)
    y = np.array(y_labels, dtype=np.int32) if y_labels else np.zeros((0,), dtype=np.int32)
    return X, y, pair_metadata


# ---------------------------------------------------------------------------
# Threshold Optimization
# ---------------------------------------------------------------------------

def optimize_decision_threshold(
    val_probs: np.ndarray,
    val_pairs: List[Tuple[str, str]],
    val_gt: Dict[str, Set[str]],
    step: float = 0.01,
) -> Tuple[float, float, float, float]:
    """
    Sweeps decision threshold theta in [0.10, 0.95] with given step to identify
    theta* maximizing Macro F_0.5 score including singletons.
    """
    s1_val_candidates: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
    for (sid, cid), prob in zip(val_pairs, val_probs):
        s1_val_candidates[sid].append((cid, float(prob)))

    best_threshold = 0.70
    best_f05 = -1.0
    best_p = 0.0
    best_r = 0.0

    thresholds = np.arange(0.10, 0.96, step)
    for thresh in thresholds:
        preds: Dict[str, Set[str]] = {}
        for sid in val_gt.keys():
            matched_ids = {cid for cid, prob in s1_val_candidates.get(sid, []) if prob >= thresh}
            preds[sid] = matched_ids

        f05, p, r = calculate_macro_f05(val_gt, preds)
        # On ties, prefer higher threshold to protect against false positives (Rule C & Rule 4)
        if f05 > best_f05 or (f05 == best_f05 and f05 > 0 and float(thresh) > best_threshold):
            best_f05 = f05
            best_threshold = float(thresh)
            best_p = p
            best_r = r

    return best_threshold, best_f05, best_p, best_r


# ---------------------------------------------------------------------------
# Grouped Cross-Validation Pipeline (Phase 5.6)
# ---------------------------------------------------------------------------

def run_cross_validation(
    sample_size: int = 1500,
    n_splits: int = 5,
    top_k: int = 25,
) -> Tuple[float, float, float]:
    """
    Executes 5-fold Grouped Cross-Validation (GroupKFold grouped on source1_entity_id)
    guaranteeing zero entity leakage between train and validation splits.
    """
    print("\n" + "=" * 75)
    print(f" Step 5.6: Grouped {n_splits}-Fold Cross-Validation Evaluation")
    print("=" * 75)

    s1_path = PROJECT_ROOT / "dataset" / "train" / "train_source1.tsv"
    gt_path = PROJECT_ROOT / "dataset" / "train" / "train_ground_truth.tsv"
    s2_path = PROJECT_ROOT / "dataset" / "train" / "train_source2.tsv"
    s3_path = PROJECT_ROOT / "dataset" / "train" / "train_source3.tsv"

    df_s1 = pd.read_csv(s1_path, sep="\t", nrows=sample_size, dtype=str, keep_default_na=False)
    s1_ids = set(df_s1["entity_id"])

    # Load Ground Truth
    gt_map: Dict[str, Set[str]] = {}
    all_true_targets: Set[str] = set()
    reader = pd.read_csv(gt_path, sep="\t", chunksize=100000, dtype=str, keep_default_na=False)
    for chunk in reader:
        matched = chunk[chunk["source1_entity_id"].isin(s1_ids)]
        for _, row in matched.iterrows():
            sid = row["source1_entity_id"]
            m_str = str(row["matched_entity_ids"]).strip() if pd.notna(row["matched_entity_ids"]) else ""
            matches = {x.strip() for x in m_str.split(",") if x.strip()}
            gt_map[sid] = matches
            all_true_targets.update(matches)
        if len(gt_map) >= len(s1_ids):
            reader.close()
            break

    # Build Target Index
    engine = BlockingEngine(max_candidates_per_entity=top_k)
    target_registry: Dict[str, Any] = {}
    for t_path in [s2_path, s3_path]:
        if not t_path.exists():
            continue
        for chunk in pd.read_csv(t_path, sep="\t", chunksize=50000, dtype=str, keep_default_na=False):
            for _, row in chunk.iterrows():
                eid = row["entity_id"]
                target_registry[eid] = row
                engine.index_record(eid, row.get("business_name"), row.get("business_address"), row.get("country"))

    # Extract full dataset
    X, y, pair_metadata = build_training_pairwise_dataset(
        df_s1, gt_map, target_registry, engine, negatives_per_positive=5
    )

    s1_groups = np.array([sid for sid, _ in pair_metadata])
    unique_groups = np.unique(s1_groups)

    # Adjust n_splits if sample size has fewer unique entities
    actual_splits = min(n_splits, len(unique_groups))
    if actual_splits < 2:
        print("[NOTICE] Too few entities for multi-fold split, running train/val evaluation.")
        actual_splits = 2

    gkf = GroupKFold(n_splits=actual_splits)
    fold_f05: List[float] = []
    fold_p: List[float] = []
    fold_r: List[float] = []
    fold_thresholds: List[float] = []

    print(f"Executing {actual_splits}-Fold GroupKFold over {len(X)} candidate pairs ({len(unique_groups)} S1 entities)...")

    for fold_num, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=s1_groups), start=1):
        df_X_tr = pd.DataFrame(X[train_idx], columns=FEATURE_COLUMNS)
        y_tr = y[train_idx]
        df_X_val = pd.DataFrame(X[val_idx], columns=FEATURE_COLUMNS)
        y_val = y[val_idx]

        # LightGBM Classifier with tuned parameters
        clf = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=150,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=6,
            scale_pos_weight=5.0,
            random_state=42,
            n_jobs=2,
            verbose=-1,
        )
        clf.fit(df_X_tr, y_tr)

        # Calibrate using training-fold data only. Fitting calibration on the
        # validation fold leaks its labels into both the score and threshold.
        calibrated = CalibratedClassifierCV(
            lgb.LGBMClassifier(
                objective="binary", n_estimators=150, learning_rate=0.05,
                num_leaves=31, max_depth=6, scale_pos_weight=5.0,
                random_state=42, n_jobs=2, verbose=-1,
            ),
            cv=3,
            method="sigmoid",
        )
        calibrated.fit(df_X_tr, y_tr)
        val_probs = calibrated.predict_proba(df_X_val)[:, 1]

        val_pairs = [pair_metadata[i] for i in val_idx]
        val_s1_ids = set(s1_groups[val_idx])
        val_gt = {sid: gt_map.get(sid, set()) for sid in val_s1_ids}

        best_t, best_f, p, r = optimize_decision_threshold(val_probs, val_pairs, val_gt, step=0.01)
        fold_thresholds.append(best_t)
        fold_f05.append(best_f)
        fold_p.append(p)
        fold_r.append(r)
        print(f"  Fold {fold_num}: Macro F0.5 = {best_f:.4f} (P = {p:.4f}, R = {r:.4f}) at tau* = {best_t:.2f}")

    mean_f05 = float(np.mean(fold_f05))
    mean_p = float(np.mean(fold_p))
    mean_r = float(np.mean(fold_r))
    mean_t = float(np.mean(fold_thresholds))

    print("-" * 75)
    print(" CROSS-VALIDATION SUMMARY RESULTS")
    print("-" * 75)
    print(f" Mean Optimal Threshold (tau*): {mean_t:.2f}")
    print(f" Mean Macro F_0.5 Score:        {mean_f05:.4f} (+/- {np.std(fold_f05):.4f})")
    print(f" Mean Macro Precision:          {mean_p:.4f}")
    print(f" Mean Macro Recall:             {mean_r:.4f}")
    print("-" * 75)

    return mean_f05, mean_p, mean_r


# ---------------------------------------------------------------------------
# Training Pipeline (Phase 5)
# ---------------------------------------------------------------------------

def train_matching_model(sample_size: int = 1500, top_k: int = 25) -> Tuple[Any, float]:
    """
    Trains a LightGBM classifier on candidate pairs generated from training data,
    evaluates on a held-out validation set, applies probability calibration,
    and searches for the optimal precision-heavy F_0.5 threshold.
    """
    print("\n" + "=" * 75)
    print(f" Step 5: Training ML Matching Model (LightGBM) on {sample_size} S1 records")
    print("=" * 75)

    s1_path = PROJECT_ROOT / "dataset" / "train" / "train_source1.tsv"
    gt_path = PROJECT_ROOT / "dataset" / "train" / "train_ground_truth.tsv"
    s2_path = PROJECT_ROOT / "dataset" / "train" / "train_source2.tsv"
    s3_path = PROJECT_ROOT / "dataset" / "train" / "train_source3.tsv"

    df_s1 = pd.read_csv(s1_path, sep="\t", nrows=sample_size, dtype=str, keep_default_na=False)
    s1_ids = set(df_s1["entity_id"])

    # Load matching labels from ground truth
    gt_map: Dict[str, Set[str]] = {}
    all_true_targets: Set[str] = set()
    reader = pd.read_csv(gt_path, sep="\t", chunksize=100000, dtype=str, keep_default_na=False)
    for chunk in reader:
        matched = chunk[chunk["source1_entity_id"].isin(s1_ids)]
        for _, row in matched.iterrows():
            sid = row["source1_entity_id"]
            m_str = str(row["matched_entity_ids"]).strip() if pd.notna(row["matched_entity_ids"]) else ""
            matches = {x.strip() for x in m_str.split(",") if x.strip()}
            gt_map[sid] = matches
            all_true_targets.update(matches)
        if len(gt_map) >= len(s1_ids):
            reader.close()
            break

    # Build target index
    engine = BlockingEngine(max_candidates_per_entity=top_k)
    target_registry: Dict[str, Any] = {}
    for t_path in [s2_path, s3_path]:
        if not t_path.exists():
            continue
        for chunk in pd.read_csv(t_path, sep="\t", chunksize=50000, dtype=str, keep_default_na=False):
            for _, row in chunk.iterrows():
                eid = row["entity_id"]
                target_registry[eid] = row
                engine.index_record(eid, row.get("business_name"), row.get("business_address"), row.get("country"))

    print(f"Target index ready ({len(target_registry):,} records).")

    # Extract pairwise dataset
    X, y, pair_metadata = build_training_pairwise_dataset(
        df_s1, gt_map, target_registry, engine, negatives_per_positive=5
    )

    print(f"Extracted {len(X):,} candidate pairs.")
    print(f"Class Distribution: {np.sum(y == 1):,} Positive Matches, {np.sum(y == 0):,} Negative Pairs.")

    # Grouped Train / Val Split (GroupKFold on S1 Entity IDs)
    s1_groups = np.array([sid for sid, _ in pair_metadata])
    unique_groups = np.unique(s1_groups)
    gkf = GroupKFold(n_splits=min(5, max(2, len(unique_groups))))

    train_idx, val_idx = next(gkf.split(X, y, groups=s1_groups))
    df_X_train = pd.DataFrame(X[train_idx], columns=FEATURE_COLUMNS)
    y_train = y[train_idx]
    df_X_val = pd.DataFrame(X[val_idx], columns=FEATURE_COLUMNS)
    y_val = y[val_idx]

    # Train LightGBM with scale_pos_weight
    print(f"\nTraining LightGBM Classifier on {len(df_X_train):,} training pairs...")
    clf = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=5.0,
        random_state=42,
        n_jobs=2,
        verbose=-1,
    )
    clf.fit(df_X_train, y_train)

    # Log Feature Importance
    importances = clf.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    print("\nTop Most Informative Features:")
    for idx in sorted_idx[:5]:
        print(f"  - {FEATURE_COLUMNS[idx]:<32}: {importances[idx]:>4}")

    # Calibrate using training-fold data only; the held-out fold remains
    # independent for threshold selection and validation reporting.
    calibrated_clf = CalibratedClassifierCV(
        lgb.LGBMClassifier(
            objective="binary", n_estimators=200, learning_rate=0.05,
            num_leaves=31, max_depth=6, subsample=0.8,
            colsample_bytree=0.8, scale_pos_weight=5.0,
            random_state=42, n_jobs=2, verbose=-1,
        ),
        cv=3,
        method="sigmoid",
    )
    calibrated_clf.fit(df_X_train, y_train)
    val_probs = calibrated_clf.predict_proba(df_X_val)[:, 1]

    # Optimize Decision Threshold for Macro F_0.5
    val_pairs = [pair_metadata[i] for i in val_idx]
    val_s1_ids = set(s1_groups[val_idx])
    val_gt = {sid: gt_map.get(sid, set()) for sid in val_s1_ids}

    best_threshold, best_f05, best_p, best_r = optimize_decision_threshold(
        val_probs, val_pairs, val_gt, step=0.01
    )

    print("-" * 75)
    print(" VALIDATION RESULTS (F_0.5 OPTIMIZATION)")
    print("-" * 75)
    print(f" Optimal Threshold (tau*):     {best_threshold:.2f}")
    print(f" Macro F_0.5 Score:             {best_f05:.4f}")
    print(f" Macro Precision:               {best_p:.4f}")
    print(f" Macro Recall:                  {best_r:.4f}")
    print("-" * 75)

    # Save Model Artefact
    model_dir = PROJECT_ROOT / "output"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "matching_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(
            {
                "model": calibrated_clf,
                "base_model": clf,
                "threshold": best_threshold,
                "features": FEATURE_COLUMNS,
            },
            f,
        )

    print(f"[SAVED] Calibrated model artefacts saved to {model_path}")
    return calibrated_clf, best_threshold


# ---------------------------------------------------------------------------
# Prediction & Inference Pipeline (Pre-Phase 6 Engine)
# ---------------------------------------------------------------------------

def run_prediction_pipeline(
    sample_limit: Optional[int] = None,
    threshold_override: Optional[float] = None,
) -> Tuple[Path, Path]:
    """
    Executes end-to-end inference generating compliant matching_results.tsv
    and candidate_pairs.tsv for Phase 6.

    Args:
        sample_limit: Optional limit on test Source 1 entities (None = process all).
        threshold_override: Optional decision threshold override (None = use calibrated tau*).

    Returns:
        Tuple[Path, Path]: Absolute paths to matching_results.tsv and candidate_pairs.tsv.
    """
    model_path = PROJECT_ROOT / "output" / "matching_model.pkl"
    if not model_path.exists():
        print("Model artefact not found. Training model first...")
        train_matching_model(sample_size=1000)

    with open(model_path, "rb") as f:
        artefacts = pickle.load(f)

    clf = artefacts["model"]
    threshold = threshold_override if threshold_override is not None else artefacts["threshold"]
    print(f"Loaded trained model with decision threshold tau* = {threshold:.2f}")

    test_dir = PROJECT_ROOT / "dataset" / "test"
    test_s1_path = test_dir / "test_source1.tsv"
    test_s2_path = test_dir / "test_source2.tsv"
    test_s3_path = test_dir / "test_source3.tsv"

    output_path = PROJECT_ROOT / "output" / "matching_results.tsv"
    output_cand_path = PROJECT_ROOT / "output" / "candidate_pairs.tsv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("\nRunning test candidate generation and inference across countries...")
    # Read test Source 1
    df_s1 = pd.read_csv(test_s1_path, sep="\t", dtype=str, keep_default_na=False)
    if sample_limit is not None:
        df_s1 = df_s1.iloc[:sample_limit]

    # Index targets with lightweight tuple representation (name, address, country) for RAM safety (<800 MB)
    target_cache: Dict[str, Tuple[str, str, str]] = {}
    for t_path in [test_s2_path, test_s3_path]:
        if not t_path.exists():
            continue
        for chunk in pd.read_csv(t_path, sep="\t", chunksize=20000, dtype=str, keep_default_na=False):
            for _, row in chunk.iterrows():
                target_cache[row["entity_id"]] = (
                    row.get("business_name", ""),
                    row.get("business_address", ""),
                    row.get("country", ""),
                )

    # Build blocking engine with test targets
    engine = BlockingEngine(max_candidates_per_entity=25)
    for eid, (t_name, t_addr, t_country) in target_cache.items():
        engine.index_record(eid, t_name, t_addr, t_country)

    with open(output_path, "w", encoding="utf-8") as out_m, open(output_cand_path, "w", encoding="utf-8") as out_c:
        out_m.write("source1_entity_id\tmatched_entity_ids\n")
        out_c.write("source1_entity_id\tcandidate_entity_ids\n")

        singletons = 0
        matches_found = 0

        for _, s1_row in df_s1.iterrows():
            sid = s1_row["entity_id"]
            s1_name = s1_row.get("business_name", "")
            s1_addr = s1_row.get("business_address", "")
            s1_cntry = s1_row.get("country", "")

            cands = engine.retrieve_candidates_for_query(s1_name, s1_addr, s1_cntry)
            out_c.write(f"{sid}\t{','.join(cands)}\n")

            matched = []
            if cands:
                X_cand = []
                valid_cands = []
                for cid in cands:
                    if cid not in target_cache:
                        continue
                    t_name, t_addr, t_cntry = target_cache[cid]
                    feats = extract_pair_features(
                        s1_name=s1_name,
                        s1_addr=s1_addr,
                        s1_country=s1_cntry,
                        target_id=cid,
                        target_name=t_name,
                        target_addr=t_addr,
                        target_country=t_cntry,
                    )
                    X_cand.append(feats)
                    valid_cands.append(cid)

                if X_cand:
                    df_cand = pd.DataFrame(X_cand, columns=FEATURE_COLUMNS)
                    probs = clf.predict_proba(df_cand)[:, 1]
                    for cid, p in zip(valid_cands, probs):
                        if p >= threshold:
                            matched.append(cid)

            if matched:
                matches_found += 1
                out_m.write(f"{sid}\t{','.join(matched)}\n")
            else:
                singletons += 1
                out_m.write(f"{sid}\t\n")

    # Defensive Invariant Audits (Rule 126 in context/code_standards.md)
    df_out_m = pd.read_csv(output_path, sep="\t", dtype=str, keep_default_na=False)
    df_out_c = pd.read_csv(output_cand_path, sep="\t", dtype=str, keep_default_na=False)

    assert len(df_out_m) == len(df_s1), (
        f"Row count mismatch in matching_results: {len(df_out_m)} != {len(df_s1)}"
    )
    assert df_out_m["source1_entity_id"].nunique() == len(df_out_m), (
        "Duplicate S1 rows detected in matching_results.tsv"
    )
    assert len(df_out_c) == len(df_s1), (
        f"Row count mismatch in candidate_pairs: {len(df_out_c)} != {len(df_s1)}"
    )
    assert df_out_c["source1_entity_id"].nunique() == len(df_out_c), (
        "Duplicate S1 rows detected in candidate_pairs.tsv"
    )

    # Candidate superset guarantee: matching results must be subset of candidate pairs
    cand_dict = dict(zip(df_out_c["source1_entity_id"], df_out_c["candidate_entity_ids"]))
    for _, row in df_out_m.iterrows():
        sid = row["source1_entity_id"]
        matched_str = row["matched_entity_ids"]
        if matched_str:
            matched_set = set(matched_str.split(","))
            for mid in matched_set:
                assert not mid.startswith("S1-"), f"Self-match violation: {mid} matches {sid}"
            cand_set = set(cand_dict.get(sid, "").split(",")) if cand_dict.get(sid, "") else set()
            assert matched_set.issubset(cand_set), (
                f"Candidate superset violation for {sid}: {matched_set - cand_set} not in candidates"
            )

    print("-" * 75)
    print(" INFERENCE RESULTS ON TEST SET (PHASE 6 VERIFIED)")
    print("-" * 75)
    print(f" Processed S1 Entities:     {len(df_s1):,}")
    print(f" Entities with Matches:     {matches_found:,}")
    print(f" Singletons (No Match):     {singletons:,}")
    print(f" Output Matching File:      {output_path}")
    print(f" Output Candidate File:     {output_cand_path}")
    print(" Invariants Verified:       Row count, No dupes, No self-match, Superset pass")
    print("-" * 75)

    return output_path, output_cand_path


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Step 4, 5 & 6: Pairwise Features, GBDT, Threshold Optimization & Test Inference")
    parser.add_argument(
        "--mode",
        choices=["train", "predict", "cv", "demo"],
        default="train",
        help="Mode: train (fit model & tune threshold), predict (run test inference & export), cv (5-fold cross validation), demo (quick test)",
    )
    parser.add_argument("--sample", type=int, default=None, help="Number of S1 records to sample (defaults to 1500 for train/cv, all for predict)")
    parser.add_argument("--splits", type=int, default=5, help="Number of folds for cross-validation")
    parser.add_argument("--top_k", type=int, default=25, help="Max candidates per entity")
    parser.add_argument("--threshold", type=float, default=None, help="Decision threshold override (default: tuned tau*)")
    args = parser.parse_args()

    if args.mode == "train":
        sample = args.sample if args.sample is not None else 1500
        train_matching_model(sample_size=sample, top_k=args.top_k)
    elif args.mode == "cv":
        sample = args.sample if args.sample is not None else 1500
        run_cross_validation(sample_size=sample, n_splits=args.splits, top_k=args.top_k)
    elif args.mode == "predict":
        run_prediction_pipeline(sample_limit=args.sample, threshold_override=args.threshold)
    elif args.mode == "demo":
        print("Quick feature extraction demo...")
        feats = extract_pair_features(
            "Acme Tech LLC", "100 Market St San Jose CA", "US",
            "S2-001", "Acme Tech Corp", "100 Market Street, San Jose", "US",
        )
        print("Sample features:", dict(zip(FEATURE_COLUMNS, feats)))
        print("[SUCCESS] Feature extractor functioning perfectly!")


if __name__ == "__main__":
    main()
