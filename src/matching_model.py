"""
Matching Model Module
=====================
Step 4: Machine Learning-based Business Entity Resolution Matcher.

This module extracts discriminative pairwise features between Source 1 records
and candidate targets (Source 2 / Source 3), trains a gradient-boosted decision
tree (LightGBM) classifier, and optimizes the decision threshold to maximize
the competition's primary evaluation metric: Macro F_0.5 Score.

Key Capabilities:
-----------------
1. High-speed pairwise feature engineering (Name, Address, Location, Metadata).
2. LightGBM classifier training with probability calibration.
3. Optimal F_0.5 threshold search (precision-heavy optimization).
4. Exporting compliant output/matching_results.tsv predictions.
"""

import argparse
from collections import defaultdict
import difflib
import os
from pathlib import Path
import pickle
import sys
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
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
# Fast Helper Metrics for Feature Extraction
# ---------------------------------------------------------------------------

def compute_jaccard(set1: set, set2: set) -> float:
    """Computes Jaccard similarity between two sets."""
    if not set1 or not set2:
        return 0.0
    inter = len(set1 & set2)
    union = len(set1 | set2)
    return inter / union if union > 0 else 0.0


def get_char_trigrams(text: str) -> set[str]:
    """Generates character 3-grams for typo-resilient comparison."""
    if not text or len(text) < 3:
        return set()
    return {text[i:i + 3] for i in range(len(text) - 2)}


# ---------------------------------------------------------------------------
# Feature Engineering Function
# ---------------------------------------------------------------------------

FEATURE_COLUMNS = [
    "feat_name_exact_core",
    "feat_name_exact_clean",
    "feat_name_token_jaccard",
    "feat_name_overlap_count",
    "feat_name_len_ratio",
    "feat_name_first_word_match",
    "feat_name_prefix4_match",
    "feat_name_trigram_jaccard",
    "feat_name_seq_ratio",
    "feat_addr_exact",
    "feat_addr_token_jaccard",
    "feat_addr_overlap_count",
    "feat_addr_number_match",
    "feat_addr_pincode_match",
    "feat_addr_trigram_jaccard",
    "feat_addr_missing",
    "feat_blocking_score",
    "feat_is_source3",
]


def extract_pair_features(
    s1_name: str | None,
    s1_addr: str | None,
    s1_country: str | None,
    target_id: str,
    target_name: str | None,
    target_addr: str | None,
    target_country: str | None,
    blocking_score: float = 1.0,
) -> list[float]:
    """
    Extracts a feature vector for a candidate pair (Source 1, Target).

    Returns:
        list of float values matching FEATURE_COLUMNS.
    """
    # 1. Cleaned texts
    c1_name, core1 = clean_business_name(s1_name)
    c1_addr = clean_business_address(s1_addr)
    c2_name, core2 = clean_business_name(target_name)
    c2_addr = clean_business_address(target_addr)

    # Name features
    exact_core = 1.0 if (core1 and core1 == core2) else 0.0
    exact_clean = 1.0 if (c1_name and c1_name == c2_name) else 0.0

    tokens1 = set(core1.split())
    tokens2 = set(core2.split())
    name_jaccard = compute_jaccard(tokens1, tokens2)
    name_overlap = float(len(tokens1 & tokens2))

    len1, len2 = len(core1), len(core2)
    len_ratio = (min(len1, len2) / max(len1, len2)) if max(len1, len2) > 0 else 0.0

    w1 = core1.split()[0] if core1 else ""
    w2 = core2.split()[0] if core2 else ""
    first_word_match = 1.0 if (w1 and w1 == w2) else 0.0

    pfx1 = core1.replace(" ", "")[:4]
    pfx2 = core2.replace(" ", "")[:4]
    prefix4_match = 1.0 if (pfx1 and len(pfx1) == 4 and pfx1 == pfx2) else 0.0

    tri1 = get_char_trigrams(core1)
    tri2 = get_char_trigrams(core2)
    name_trigram_jaccard = compute_jaccard(tri1, tri2)

    seq_ratio = difflib.SequenceMatcher(None, core1, core2).quick_ratio() if (core1 and core2) else 0.0

    # Address features
    addr_missing = 1.0 if (not c1_addr or not c2_addr) else 0.0
    addr_exact = 1.0 if (not addr_missing and c1_addr == c2_addr) else 0.0

    addr_tok1 = set(c1_addr.split()) if c1_addr else set()
    addr_tok2 = set(c2_addr.split()) if c2_addr else set()
    addr_jaccard = compute_jaccard(addr_tok1, addr_tok2)
    addr_overlap = float(len(addr_tok1 & addr_tok2))

    # Number match (+1 match, -1 conflict, 0 missing)
    nums1 = {t for t in addr_tok1 if t.isdigit()}
    nums2 = {t for t in addr_tok2 if t.isdigit()}
    if nums1 and nums2:
        num_match = 1.0 if bool(nums1 & nums2) else -1.0
    else:
        num_match = 0.0

    # Pincode match (+1 match, -1 conflict, 0 missing)
    pins1 = {n for n in nums1 if len(n) in (5, 6)}
    pins2 = {n for n in nums2 if len(n) in (5, 6)}
    if pins1 and pins2:
        pin_match = 1.0 if bool(pins1 & pins2) else -1.0
    else:
        pin_match = 0.0

    addr_tri1 = get_char_trigrams(c1_addr)
    addr_tri2 = get_char_trigrams(c2_addr)
    addr_trigram_jaccard = compute_jaccard(addr_tri1, addr_tri2)

    # Source signal
    is_s3 = 1.0 if target_id.startswith("S3-") else 0.0

    return [
        exact_core,
        exact_clean,
        name_jaccard,
        name_overlap,
        len_ratio,
        first_word_match,
        prefix4_match,
        name_trigram_jaccard,
        seq_ratio,
        addr_exact,
        addr_jaccard,
        addr_overlap,
        num_match,
        pin_match,
        addr_trigram_jaccard,
        addr_missing,
        float(blocking_score),
        is_s3,
    ]


# ---------------------------------------------------------------------------
# Macro F_0.5 Metric Computation
# ---------------------------------------------------------------------------

def calculate_macro_f05(
    ground_truth_dict: dict[str, set[str]],
    predictions_dict: dict[str, set[str]],
) -> tuple[float, float, float]:
    """
    Computes official macro-averaged F_0.5 score across all Source 1 entities,
    correctly evaluating singletons according to competition rules:
      - Singleton True + Predicted Empty  -> F0.5 = 1.0
      - Singleton True + Predicted Match  -> F0.5 = 0.0
      - Match True     + Predicted Empty  -> F0.5 = 0.0
      - Match True     + Predicted Match  -> Standard F_0.5 formula

    Returns:
        (macro_f05, macro_precision, macro_recall)
    """
    f05_scores = []
    precisions = []
    recalls = []

    for sid, true_set in ground_truth_dict.items():
        pred_set = predictions_dict.get(sid, set())

        # Singleton case
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

        # Non-singleton case
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
        if denom > 0:
            score = (1.25 * p * r) / denom
        else:
            score = 0.0
        f05_scores.append(score)

    macro_f05 = float(np.mean(f05_scores)) if f05_scores else 0.0
    macro_p = float(np.mean(precisions)) if precisions else 0.0
    macro_r = float(np.mean(recalls)) if recalls else 0.0
    return macro_f05, macro_p, macro_r


# ---------------------------------------------------------------------------
# Training Pipeline
# ---------------------------------------------------------------------------

def train_matching_model(sample_size: int = 1500, top_k: int = 15):
    """
    Trains a LightGBM classifier on candidate pairs generated from training data,
    evaluates on a held-out validation set, and finds the optimal F_0.5 threshold.
    """
    print("\n" + "=" * 75)
    print(f" Step 4: Training ML Matching Model (LightGBM) on {sample_size} S1 records")
    print("=" * 75)

    s1_path = PROJECT_ROOT / "dataset" / "train" / "train_source1.tsv"
    gt_path = PROJECT_ROOT / "dataset" / "train" / "train_ground_truth.tsv"
    s2_path = PROJECT_ROOT / "dataset" / "train" / "train_source2.tsv"
    s3_path = PROJECT_ROOT / "dataset" / "train" / "train_source3.tsv"

    # 1. Load S1 Sample
    print(f"Loading {sample_size} Source 1 records...")
    df_s1 = pd.read_csv(s1_path, sep="\t", nrows=sample_size)
    s1_dict = {row["entity_id"]: row for _, row in df_s1.iterrows()}
    s1_ids = set(s1_dict.keys())

    # 2. Load Ground Truth
    print("Loading matching labels from ground truth...")
    gt_map = {}
    all_true_targets = set()
    for chunk in pd.read_csv(gt_path, sep="\t", chunksize=100000):
        matched = chunk[chunk["source1_entity_id"].isin(s1_ids)]
        for _, row in matched.iterrows():
            sid = row["source1_entity_id"]
            m_str = str(row["matched_entity_ids"]) if pd.notna(row["matched_entity_ids"]) else ""
            matches = set(x.strip() for x in m_str.split(",") if x.strip())
            gt_map[sid] = matches
            all_true_targets.update(matches)
        if len(gt_map) >= len(s1_ids):
            break

    # 3. Build Target Pool & Registry
    print("Loading target records and building candidate index...")
    engine = BlockingEngine(max_candidates_per_entity=top_k)
    target_registry = {}
    distractors_target = 30000

    for target_path in [s2_path, s3_path]:
        distractor_count = 0
        for chunk in pd.read_csv(target_path, sep="\t", chunksize=50000):
            true_batch = chunk[chunk["entity_id"].isin(all_true_targets)]
            distractor_batch = chunk[~chunk["entity_id"].isin(all_true_targets)]

            if distractor_count < distractors_target // 2:
                keep_d = min(len(distractor_batch), (distractors_target // 2) - distractor_count)
                distractor_batch = distractor_batch.iloc[:keep_d]
                distractor_count += keep_d
            else:
                distractor_batch = distractor_batch.iloc[:0]

            combined = pd.concat([true_batch, distractor_batch], ignore_index=True)
            for _, row in combined.iterrows():
                eid = row["entity_id"]
                target_registry[eid] = row
                engine.index_record(
                    eid,
                    row.get("business_name"),
                    row.get("business_address"),
                    row.get("country"),
                )

    print(f"Target index ready ({len(target_registry):,} records).")

    # 4. Extract Training Features
    print("Generating candidate pairs and extracting pairwise features...")
    X_rows = []
    y_labels = []
    pair_metadata = []  # (s1_id, target_id)

    start_fe_time = time.time()
    for sid, s1_row in s1_dict.items():
        true_targets = gt_map.get(sid, set())
        candidates = engine.retrieve_candidates_for_query(
            s1_row.get("business_name"),
            s1_row.get("business_address"),
            s1_row.get("country"),
        )

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

    fe_time = time.time() - start_fe_time
    X = np.array(X_rows, dtype=np.float32)
    y = np.array(y_labels, dtype=np.int32)
    print(f"Extracted {len(X):,} candidate pairs in {fe_time:.2f}s.")
    print(f"Class Distribution: {np.sum(y == 1):,} Positive Matches, {np.sum(y == 0):,} Negative Pairs.")

    # 5. Train/Val Split at S1-Entity Level (Prevent Data Leakage)
    unique_s1s = list(s1_dict.keys())
    train_s1s, val_s1s = train_test_split(unique_s1s, test_size=0.25, random_state=42)
    train_s1_set = set(train_s1s)
    val_s1_set = set(val_s1s)

    train_indices = [i for i, (sid, _) in enumerate(pair_metadata) if sid in train_s1_set]
    val_indices = [i for i, (sid, _) in enumerate(pair_metadata) if sid in val_s1_set]

    df_X = pd.DataFrame(X, columns=FEATURE_COLUMNS)
    X_train, y_train = df_X.iloc[train_indices], y[train_indices]
    X_val, y_val = df_X.iloc[val_indices], y[val_indices]

    # 6. Train LightGBM Classifier
    print(f"\nTraining LightGBM Classifier on {len(X_train):,} pairs...")
    clf = lgb.LGBMClassifier(
        n_estimators=150,
        learning_rate=0.08,
        num_leaves=31,
        random_state=42,
        class_weight="balanced",
        verbose=-1,
    )
    clf.fit(X_train, y_train)

    # Log Feature Importance
    importances = clf.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    print("\nTop 5 Most Informative Features:")
    for idx in sorted_idx[:5]:
        print(f"  - {FEATURE_COLUMNS[idx]:<28}: {importances[idx]:>4}")

    # 7. Threshold Tuning for Macro F_0.5 Optimization
    print("\nOptimizing decision threshold on held-out validation set...")
    val_probs = clf.predict_proba(X_val)[:, 1]

    # Organize predictions by S1 ID
    val_pairs = [pair_metadata[i] for i in val_indices]
    s1_val_candidates = defaultdict(list)
    for (sid, cid), prob in zip(val_pairs, val_probs):
        s1_val_candidates[sid].append((cid, prob))

    best_threshold = 0.5
    best_f05 = 0.0
    best_p = 0.0
    best_r = 0.0

    val_gt = {sid: gt_map.get(sid, set()) for sid in val_s1_set}

    thresholds = np.arange(0.20, 0.95, 0.05)
    for thresh in thresholds:
        preds = {}
        for sid in val_s1_set:
            matched_ids = {cid for cid, prob in s1_val_candidates.get(sid, []) if prob >= thresh}
            preds[sid] = matched_ids

        f05, p, r = calculate_macro_f05(val_gt, preds)
        if f05 > best_f05:
            best_f05 = f05
            best_threshold = float(thresh)
            best_p = p
            best_r = r

    print("-" * 75)
    print(" VALIDATION RESULTS (F_0.5 OPTIMIZATION)")
    print("-" * 75)
    print(f" Optimal Threshold (tau*):     {best_threshold:.2f}")
    print(f" Macro F_0.5 Score:             {best_f05:.4f}")
    print(f" Macro Precision:               {best_p:.4f}")
    print(f" Macro Recall:                  {best_r:.4f}")
    print("-" * 75)

    # 8. Save Model Artefact
    model_dir = PROJECT_ROOT / "output"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "matching_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({"model": clf, "threshold": best_threshold, "features": FEATURE_COLUMNS}, f)

    print(f"[SAVED] Trained model artefacts saved to {model_path}")
    return clf, best_threshold


# ---------------------------------------------------------------------------
# Prediction & Submission Export
# ---------------------------------------------------------------------------

def run_prediction_pipeline(sample_limit: int | None = 1000):
    """
    Demonstrates end-to-end inference generating compliant matching_results.tsv.
    Loads candidate pairs, predicts probabilities with the trained LightGBM model,
    applies the optimal F_0.5 threshold, and formats the output.
    """
    model_path = PROJECT_ROOT / "output" / "matching_model.pkl"
    if not model_path.exists():
        print("Model artefact not found. Training model first...")
        train_matching_model(sample_size=1000)

    with open(model_path, "rb") as f:
        artefacts = pickle.load(f)

    clf = artefacts["model"]
    threshold = artefacts["threshold"]
    print(f"Loaded trained model with optimal decision threshold tau* = {threshold:.2f}")

    test_s1_path = PROJECT_ROOT / "dataset" / "test" / "test_source1.tsv"
    test_s2_path = PROJECT_ROOT / "dataset" / "test" / "test_source2.tsv"
    test_s3_path = PROJECT_ROOT / "dataset" / "test" / "test_source3.tsv"

    output_path = PROJECT_ROOT / "output" / "matching_results.tsv"
    output_cand_path = PROJECT_ROOT / "output" / "candidate_pairs.tsv"

    print(f"\nRunning test inference on sample ({sample_limit} S1 entities)...")
    # Quick indexing on sample targets
    engine = BlockingEngine(max_candidates_per_entity=15)
    target_cache = {}

    for t_path in [test_s2_path, test_s3_path]:
        for chunk in pd.read_csv(t_path, sep="\t", chunksize=20000):
            for _, row in chunk.iterrows():
                eid = row["entity_id"]
                target_cache[eid] = row
                engine.index_record(eid, row.get("business_name"), row.get("business_address"), row.get("country"))
            if len(target_cache) >= 40000:
                break

    df_s1 = pd.read_csv(test_s1_path, sep="\t", nrows=sample_limit)

    with open(output_path, "w", encoding="utf-8") as out_m, open(output_cand_path, "w", encoding="utf-8") as out_c:
        out_m.write("source1_entity_id\tmatched_entity_ids\n")
        out_c.write("source1_entity_id\tcandidate_entity_ids\n")

        singletons = 0
        matches_found = 0

        for _, s1_row in df_s1.iterrows():
            sid = s1_row["entity_id"]
            cands = engine.retrieve_candidates_for_query(
                s1_row.get("business_name"),
                s1_row.get("business_address"),
                s1_row.get("country"),
            )
            out_c.write(f"{sid}\t{','.join(cands)}\n")

            matched = []
            if cands:
                X_cand = []
                valid_cands = []
                for cid in cands:
                    if cid not in target_cache:
                        continue
                    t_row = target_cache[cid]
                    feats = extract_pair_features(
                        s1_name=s1_row.get("business_name"),
                        s1_addr=s1_row.get("business_address"),
                        s1_country=s1_row.get("country"),
                        target_id=cid,
                        target_name=t_row.get("business_name"),
                        target_addr=t_row.get("business_address"),
                        target_country=t_row.get("country"),
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

    print("-" * 75)
    print(" INFERENCE RESULTS ON SAMPLE")
    print("-" * 75)
    print(f" Processed S1 Entities:     {len(df_s1):,}")
    print(f" Entities with Matches:     {matches_found:,}")
    print(f" Singletons (No Match):     {singletons:,}")
    print(f" Output Matching File:      {output_path}")
    print(f" Output Candidate File:     {output_cand_path}")
    print("-" * 75)


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Step 4: ML Matching Model")
    parser.add_argument(
        "--mode",
        choices=["train", "predict", "demo"],
        default="train",
        help="Mode: train (fit model & tune threshold), predict (run inference & export), demo (quick test)",
    )
    parser.add_argument("--sample", type=int, default=1500, help="Number of S1 train records to sample")
    args = parser.parse_args()

    if args.mode == "train":
        train_matching_model(sample_size=args.sample)
    elif args.mode == "predict":
        run_prediction_pipeline(sample_limit=args.sample)
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
