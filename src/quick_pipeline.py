"""Train the current repository matcher on a bounded sample, then score all test S1s."""
from __future__ import annotations

import csv
from concurrent.futures import ProcessPoolExecutor
import gc
import os
import pickle
import random
import re
import sqlite3
import time
from collections import defaultdict
from pathlib import Path
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.blocking import BlockingEngine
from src.data_cleaning import clean_business_name, clean_business_address, clean_country
from src.matching_model import (
    FEATURE_COLUMNS, extract_pair_features, optimize_decision_threshold,
)

TRAIN = ROOT / "dataset" / "train"
TEST = ROOT / "dataset" / "test"
OUT = ROOT / "output"
DBPATH = OUT / "quick_target_index.sqlite"
TOP_K = 25
NONWORD = re.compile(r"[^a-z0-9]+")
POSTAL_RE = re.compile(r"(?<!\d)\d{5,6}(?!\d)")
LEGAL = {"inc", "incorporated", "corp", "corporation", "co", "company", "ltd", "limited",
         "llc", "llp", "pvt", "private", "plc", "gmbh", "sarl", "sas", "sasu", "eurl", "snc", "sa"}

_WORKER_DB = None
_WORKER_MODEL = None
_WORKER_THRESHOLD = None


def init_score_worker(db_path: str, model_path: str, threshold: float):
    global _WORKER_DB, _WORKER_MODEL, _WORKER_THRESHOLD
    _WORKER_DB = sqlite3.connect(db_path, timeout=60)
    _WORKER_DB.execute("PRAGMA query_only=ON")
    _WORKER_DB.execute("PRAGMA cache_size=-8192")
    with open(model_path, "rb") as f:
        _WORKER_MODEL = pickle.load(f)["model"]
    _WORKER_THRESHOLD = threshold


def score_source1_batch(source1_rows):
    scored = []
    feature_rows = []
    feature_owners = []
    for sid, name, address, country in source1_rows:
        cn, core, pfx, postal = norm_parts(name, address, country)
        rows = []
        seen_ids = set()
        if core:
            exact_rows = _WORKER_DB.execute(
                "SELECT id,name,address,country FROM target WHERE country=? AND core=? LIMIT ?",
                (cn, core, TOP_K),
            ).fetchall()
            rows.extend(exact_rows)
            seen_ids.update(r[0] for r in exact_rows)
        if len(rows) < TOP_K and pfx:
            prefix_rows = _WORKER_DB.execute(
                "SELECT id,name,address,country FROM target WHERE country=? AND pfx=? LIMIT ?",
                (cn, pfx, TOP_K),
            ).fetchall()
            for row in prefix_rows:
                if row[0] not in seen_ids:
                    rows.append(row)
                    seen_ids.add(row[0])
                    if len(rows) == TOP_K:
                        break
        if len(rows) < TOP_K and postal:
            postal_rows = _WORKER_DB.execute(
                "SELECT id,name,address,country FROM target WHERE country=? AND postal=? LIMIT ?",
                (cn, postal, TOP_K),
            ).fetchall()
            for row in postal_rows:
                if row[0] not in seen_ids:
                    rows.append(row)
                    seen_ids.add(row[0])
                    if len(rows) == TOP_K:
                        break
        cids = [r[0] for r in rows]
        owner_index = len(scored)
        scored.append([sid, cids, []])
        for eid, target_name, target_address, target_country in rows:
            feature_rows.append(extract_pair_features(
                name, address, country, eid, target_name, target_address, target_country
            ))
            feature_owners.append((owner_index, eid))
    if feature_rows:
        probabilities = _WORKER_MODEL.predict_proba(
            pd.DataFrame(feature_rows, columns=FEATURE_COLUMNS)
        )[:, 1]
        for (owner, eid), probability in zip(feature_owners, probabilities):
            if probability >= _WORKER_THRESHOLD:
                scored[owner][2].append(eid)
    return scored


def train_model(sample_s1: int = 12000, random_targets: int = 40000):
    print("Loading a representative training sample and labels...", flush=True)
    rng = random.Random(731)
    s1_rows = []
    with (TRAIN / "train_source1.tsv").open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        reservoir = []
        for i, row in enumerate(reader):
            if i < sample_s1:
                reservoir.append(row)
            else:
                j = rng.randrange(i + 1)
                if j < sample_s1:
                    reservoir[j] = row
        s1_rows = reservoir
    s1map = {r["entity_id"]: r for r in s1_rows}
    gt = {}
    with (TRAIN / "train_ground_truth.tsv").open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            sid = row["source1_entity_id"]
            if sid in s1map:
                gt[sid] = {x for x in row.get("matched_entity_ids", "").split(",") if x}
    positive_ids = set().union(*gt.values()) if gt else set()

    print(f"Sampled {len(s1_rows):,} S1 rows; reading labelled and distractor targets...", flush=True)
    targets = {}
    sampled = []
    seen = 0
    for filename in ("train_source2.tsv", "train_source3.tsv"):
        with (TRAIN / filename).open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                eid = row["entity_id"]
                if eid in positive_ids:
                    targets[eid] = row
                seen += 1
                # Reservoir-sample generic in-country negatives without retaining the full target set.
                if len(sampled) < random_targets:
                    sampled.append(row)
                else:
                    j = rng.randrange(seen)
                    if j < random_targets:
                        sampled[j] = row
    for row in sampled:
        targets.setdefault(row["entity_id"], row)
    del sampled
    gc.collect()

    engine = BlockingEngine(max_candidates_per_entity=TOP_K)
    by_country = defaultdict(list)
    for eid, row in targets.items():
        engine.index_record(eid, row["business_name"], row["business_address"], row["country"])
        by_country[clean_country(row["country"])].append(eid)

    # Split Source 1 entities before pair construction. Validation must use
    # naturally retrieved blocking candidates, without inserting known truths.
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    row_ids = np.arange(len(s1_rows))
    train_rows, val_rows = next(splitter.split(row_ids, groups=row_ids))
    train_sids = {s1_rows[i]["entity_id"] for i in train_rows}
    val_sids = {s1_rows[i]["entity_id"] for i in val_rows}

    X, y, pairs = [], [], []
    retrieved_true = 0
    total_true = 0
    for row in s1_rows:
        sid = row["entity_id"]
        true_ids = gt.get(sid, set())
        cands = engine.retrieve_candidates_for_query(
            row["business_name"], row["business_address"], row["country"]
        )
        total_true += len(true_ids)
        retrieved_true += len(set(cands) & true_ids)
        for cid in cands:
            target = targets.get(cid)
            if target is None:
                continue
            X.append(extract_pair_features(row["business_name"], row["business_address"], row["country"],
                                           cid, target["business_name"], target["business_address"], target["country"]))
            y.append(int(cid in true_ids))
            pairs.append((sid, cid))
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int32)
    groups = np.asarray([sid for sid, _ in pairs])
    train_mask = np.asarray([sid in train_sids for sid in groups])
    val_mask = ~train_mask
    print(
        f"Training pairs: {len(X):,}; positives: {int(y.sum()):,}; "
        f"S1: {len(s1_rows):,}; blocking recall: {retrieved_true / max(total_true, 1):.4f}",
        flush=True,
    )
    if len(np.unique(y)) < 2:
        raise RuntimeError("Training sample did not contain both positive and negative pairs")

    tr, va = np.flatnonzero(train_mask), np.flatnonzero(val_mask)
    if not len(tr) or not len(va) or len(np.unique(y[tr])) < 2:
        raise RuntimeError("Grouped split did not produce both classes in training and validation")
    model = lgb.LGBMClassifier(objective="binary", n_estimators=180, learning_rate=0.05,
                               num_leaves=31, max_depth=6, scale_pos_weight=4.0,
                               random_state=42, n_jobs=2, verbose=-1)
    model.fit(pd.DataFrame(X[tr], columns=FEATURE_COLUMNS), y[tr])
    Xv = pd.DataFrame(X[va], columns=FEATURE_COLUMNS)
    # Validation labels are used only for threshold selection, never calibration.
    probs = model.predict_proba(Xv)[:, 1]
    valpairs = [pairs[i] for i in va]
    val_gt = {sid: gt.get(sid, set()) for sid in val_sids}
    threshold, score, precision, recall = optimize_decision_threshold(probs, valpairs, val_gt, step=0.01)
    print(f"Validation sample: macro F0.5={score:.4f}, P={precision:.4f}, R={recall:.4f}, threshold={threshold:.2f}", flush=True)
    with (OUT / "matching_model.pkl").open("wb") as f:
        pickle.dump({"model": model, "base_model": model, "threshold": threshold,
                     "features": FEATURE_COLUMNS}, f)
    del targets, engine, X, y, groups
    gc.collect()
    return model, threshold


def norm_parts(name: str, address: str, country: str):
    # Cheap, high-recall index normalization; full latest-repo cleaning is applied
    # to the small retrieved candidate set during feature extraction.
    raw = NONWORD.sub(" ", name.lower().replace("&", " and "))
    parts = raw.split()
    while parts and parts[-1] in LEGAL:
        parts.pop()
    core = " ".join(parts)
    compact = core.replace(" ", "")
    postal = POSTAL_RE.search(address)
    return clean_country(country), core, compact[:4] if len(compact) >= 4 else "", postal.group(0) if postal else ""


def build_test_index(db):
    db.executescript("DROP TABLE IF EXISTS target; CREATE TABLE target(id TEXT PRIMARY KEY, country TEXT, core TEXT, pfx TEXT, postal TEXT, name TEXT, address TEXT);")
    add = "INSERT OR REPLACE INTO target VALUES (?,?,?,?,?,?,?)"
    buf = []
    total = 0
    started = time.time()
    for filename in ("test_source2.tsv", "test_source3.tsv"):
        with (TEST / filename).open(encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f, delimiter="\t"):
                name, address, country = r.get("business_name", ""), r.get("business_address", ""), r.get("country", "")
                cn, core, pfx, postal = norm_parts(name, address, country)
                buf.append((r["entity_id"], cn, core, pfx, postal, name, address))
                total += 1
                if len(buf) == 5000:
                    db.executemany(add, buf); db.commit(); buf.clear()
                    if total % 250000 == 0:
                        print(f"Indexed {total:,} test targets ({time.time()-started:.0f}s)", flush=True)
    if buf:
        db.executemany(add, buf); db.commit()
    print("Creating disk indexes...", flush=True)
    db.execute("CREATE INDEX target_core ON target(country,core)")
    db.execute("CREATE INDEX target_pfx ON target(country,pfx)")
    db.execute("CREATE INDEX target_postal ON target(country,postal)")
    db.commit()
    print(f"Target index ready: {total:,} records, {time.time()-started:.0f}s", flush=True)


def predict_all(model, threshold):
    db = sqlite3.connect(DBPATH)
    db.execute("PRAGMA journal_mode=OFF")
    db.execute("PRAGMA synchronous=OFF")
    db.execute("PRAGMA temp_store=FILE")
    db.execute("PRAGMA cache_size=-32768")
    existing_indexes = {row[1] for row in db.execute("PRAGMA index_list('target')")} if DBPATH.exists() else set()
    if "target_core" not in existing_indexes or "target_pfx" not in existing_indexes:
        db.close()
        if DBPATH.exists():
            DBPATH.unlink()
        db = sqlite3.connect(DBPATH)
        db.execute("PRAGMA journal_mode=OFF")
        db.execute("PRAGMA synchronous=OFF")
        db.execute("PRAGMA temp_store=FILE")
        db.execute("PRAGMA cache_size=-32768")
        build_test_index(db)
    else:
        print("Reusing completed on-disk target index", flush=True)

    mpath, cpath = OUT / "matching_results.tsv", OUT / "candidate_pairs.tsv"
    total = matched = 0
    start = time.time()
    worker_count = min(8, max(2, os.cpu_count() or 4))
    print(f"Scoring with {worker_count} workers", flush=True)
    with mpath.open("w", encoding="utf-8", newline="") as fm, cpath.open("w", encoding="utf-8", newline="") as fc:
        mw, cw = csv.writer(fm, delimiter="\t", lineterminator="\n"), csv.writer(fc, delimiter="\t", lineterminator="\n")
        mw.writerow(["source1_entity_id", "matched_entity_ids"])
        cw.writerow(["source1_entity_id", "candidate_entity_ids"])
        db.close()
        pending = []
        with ProcessPoolExecutor(max_workers=worker_count, initializer=init_score_worker,
                                 initargs=(str(DBPATH), str(OUT / "matching_model.pkl"), threshold)) as pool:
            for chunk in pd.read_csv(TEST / "test_source1.tsv", sep="\t", chunksize=2000,
                                     dtype=str, keep_default_na=False):
                rows = list(chunk.itertuples(index=False, name=None))
                tasks = [rows[i:i+250] for i in range(0, len(rows), 250)]
                for batch_result in pool.map(score_source1_batch, tasks):
                    for sid, cids, mids in batch_result:
                        cw.writerow([sid, ",".join(cids)])
                        mw.writerow([sid, ",".join(mids)])
                        total += 1
                        matched += bool(mids)
                        if total % 10000 == 0:
                            fm.flush(); fc.flush()
                            print(f"Scored {total:,}/{1732544:,} S1; {matched:,} with predictions ({time.time()-start:.0f}s)", flush=True)
    db.close()
    print(f"Prediction complete: {total:,} rows, {matched:,} non-empty ({time.time()-start:.0f}s)", flush=True)


def main():
    OUT.mkdir(exist_ok=True)
    model_path = OUT / "matching_model.pkl"
    try:
        with model_path.open("rb") as f:
            saved = pickle.load(f)
        if saved.get("features") == FEATURE_COLUMNS:
            model, threshold = saved["model"], float(saved["threshold"])
            print(f"Using newly trained repository model; threshold={threshold:.2f}", flush=True)
        else:
            model, threshold = train_model()
    except (OSError, EOFError, pickle.UnpicklingError):
        model, threshold = train_model()
    predict_all(model, threshold)


if __name__ == "__main__":
    main()
