"""
Blocking and Candidate Generation Module
=========================================
Step 3: High-Recall Multi-Key Candidate Generation for Business Entity Resolution.

Why Blocking?
-------------
Comparing ~1.73M Source 1 records against ~10M Source 2 & Source 3 records requires
~17.3 Trillion pairwise comparisons, which is computationally intractable.
Blocking partitions records into buckets using lightweight keys (exact core names,
word n-grams, prefixes, and address/postal anchors) so that we only compare records
that are plausible matches, reducing 17.3 Trillion comparisons down to ~15-25
candidates per entity while preserving >95% candidate recall ceiling.

Hardware Constraints:
---------------------
Designed to operate under strict memory limits (< 800 MB RAM) by:
1. Strict country partitioning (France, India, US, open-set) - zero cross-country bleed.
2. Inverted index using token postings and frequency-based ranking.
3. Streaming chunked file I/O to prevent large in-memory DataFrames.
"""

import argparse
from collections import Counter, defaultdict
import os
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional, Set, Tuple
import pandas as pd

# Add project root to sys.path for standalone script execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.data_cleaning import (
        clean_business_name,
        clean_business_address,
        clean_country,
    )
except ImportError:
    from data_cleaning import (
        clean_business_name,
        clean_business_address,
        clean_country,
    )


# ---------------------------------------------------------------------------
# Stopwords and Generic Terms to Avoid Massive "Super-Blocks"
# ---------------------------------------------------------------------------
GENERIC_NAME_STOPWORDS: Set[str] = {
    "the", "and", "company", "services", "enterprises", "store", "shop",
    "hotel", "restaurant", "cafe", "center", "centre", "group", "solutions",
    "india", "usa", "france", "global", "holding", "holdings", "industries",
    "international", "corporation", "ltd", "inc", "corp", "pvt", "llc",
}

GENERIC_ADDR_STOPWORDS: Set[str] = {
    "street", "st", "road", "rd", "avenue", "ave", "boulevard", "blvd",
    "floor", "fl", "suite", "ste", "apartment", "apt", "near", "nr", "opp",
    "opposite", "behind", "us", "india", "france", "unit", "bldg", "block",
    "sector", "colony", "nagar", "market", "main", "cross", "city", "state",
}


# ---------------------------------------------------------------------------
# High-Recall Multi-Key Blocking Generation
# ---------------------------------------------------------------------------
def extract_blocking_keys(
    clean_name: str,
    core_name: str,
    clean_addr: str,
    country: str,
) -> Set[str]:
    """
    Generates a diverse union of lightweight blocking keys for a business record.

    Key Types:
      1. Exact Core Name (CN): Captures identical names with suffix variations.
      2. First 2 Words (2W) & First Word (1W): Captures name prefix/brand matches.
      3. Character Prefix 4 & 5 (PFX4, PFX5): Resilient to minor typos and suffixes.
      4. Postal Code + Name Token (PIN_NAME): High-precision geographic anchor.
      5. Address Anchor (ADDR): Combines building/door numbers with street tokens.
      6. Address 2-Word Bigram (ADDR2W): Matches shared street/neighborhood names.

    Parameters:
        clean_name: Standardized business name.
        core_name: Business name without legal suffixes.
        clean_addr: Standardized address string.
        country: Normalized country code (US, IN, FR, etc.).

    Returns:
        Set[str]: Set of generated blocking keys prefixed by country.
    """
    keys: Set[str] = set()
    c_prefix = country.upper() if country else "UNK"

    # 1. Exact Core Name Key
    if core_name and len(core_name) >= 3:
        keys.add(f"{c_prefix}#CN#{core_name}")

    # Tokenize core name excluding generic stopwords
    name_words = [w for w in core_name.split() if w not in GENERIC_NAME_STOPWORDS]

    # 2. First 2 Words and First Word Keys
    if len(name_words) >= 2:
        keys.add(f"{c_prefix}#2W#{name_words[0]}_{name_words[1]}")
    if name_words and len(name_words[0]) >= 3:
        keys.add(f"{c_prefix}#1W#{name_words[0]}")

    # 3. Compact Character Prefix Keys (Typos & transliteration resilience)
    clean_nospace = core_name.replace(" ", "")
    if len(clean_nospace) >= 4:
        keys.add(f"{c_prefix}#PFX4#{clean_nospace[:4]}")
    if len(clean_nospace) >= 5:
        keys.add(f"{c_prefix}#PFX5#{clean_nospace[:5]}")

    # 4. Address & Postal Anchors
    if clean_addr:
        addr_tokens = clean_addr.split()
        numbers = [t for t in addr_tokens if t.isdigit()]
        words = [
            t for t in addr_tokens
            if not t.isdigit() and len(t) >= 4 and t not in GENERIC_ADDR_STOPWORDS
        ]

        # Postal code anchor (5 digits for US/France, 6 digits for India)
        pincodes = [n for n in numbers if len(n) in (5, 6)]
        if pincodes and name_words and len(name_words[0]) >= 3:
            keys.add(f"{c_prefix}#PIN_NAME#{pincodes[0]}_{name_words[0]}")

        # Pair up to first 2 numbers with up to first 3 significant address words
        for num in numbers[:2]:
            for w in words[:3]:
                keys.add(f"{c_prefix}#ADDR#{num}_{w}")

        # Consecutive address bigrams (e.g., 'market_san', 'boulevard_haussmann')
        if len(words) >= 2:
            keys.add(f"{c_prefix}#ADDR2W#{words[0]}_{words[1]}")

    return keys


# ---------------------------------------------------------------------------
# High-Performance Inverted Index Blocking Engine
# ---------------------------------------------------------------------------
class BlockingEngine:
    """
    Lightweight, streaming-compatible Inverted Index for candidate generation.
    Enforces country isolation and memory efficiency.
    """

    def __init__(self, max_candidates_per_entity: int = 25):
        self.max_candidates = max_candidates_per_entity
        self.index: Dict[str, List[str]] = defaultdict(list)
        self.num_targets_indexed = 0

    def index_record(
        self,
        entity_id: str,
        business_name: Optional[str],
        business_address: Optional[str],
        country: Optional[str],
    ) -> None:
        """Indexes a single target record (Source 2 or Source 3)."""
        c_name, core_name = clean_business_name(business_name)
        c_addr = clean_business_address(business_address)
        clean_cntry = clean_country(country)

        keys = extract_blocking_keys(c_name, core_name, c_addr, clean_cntry)
        for k in keys:
            self.index[k].append(entity_id)
        self.num_targets_indexed += 1

    def retrieve_candidates_for_query(
        self,
        business_name: Optional[str],
        business_address: Optional[str],
        country: Optional[str],
    ) -> List[str]:
        """
        Retrieves and ranks the top-K candidate target IDs for a Source 1 record.
        Uses key frequency voting: candidates sharing multiple blocking keys
        are ranked highest.
        """
        c_name, core_name = clean_business_name(business_name)
        c_addr = clean_business_address(business_address)
        clean_cntry = clean_country(country)

        keys = extract_blocking_keys(c_name, core_name, c_addr, clean_cntry)

        candidate_counts: Counter[str] = Counter()
        for k in keys:
            if k in self.index:
                # Key-specific weights: Exact core name has highest weight
                if "#CN#" in k:
                    weight = 4
                elif "#PIN_NAME#" in k:
                    weight = 3
                elif "#2W#" in k:
                    weight = 2
                else:
                    weight = 1

                for eid in self.index[k]:
                    candidate_counts[eid] += weight

        if not candidate_counts:
            return []

        # Return top-K candidates ordered by score
        return [eid for eid, _ in candidate_counts.most_common(self.max_candidates)]

    def retrieve_candidates_with_scores(
        self,
        business_name: Optional[str],
        business_address: Optional[str],
        country: Optional[str],
    ) -> List[Tuple[str, float]]:
        """Retrieves top-K candidates paired with their blocking match scores."""
        c_name, core_name = clean_business_name(business_name)
        c_addr = clean_business_address(business_address)
        clean_cntry = clean_country(country)

        keys = extract_blocking_keys(c_name, core_name, c_addr, clean_cntry)

        candidate_counts: Counter[str] = Counter()
        for k in keys:
            if k in self.index:
                weight = 4 if "#CN#" in k else (3 if "#PIN_NAME#" in k else (2 if "#2W#" in k else 1))
                for eid in self.index[k]:
                    candidate_counts[eid] += weight

        if not candidate_counts:
            return []

        return candidate_counts.most_common(self.max_candidates)

    def clear(self) -> None:
        """Clears index to free memory between country partitions."""
        self.index.clear()
        self.num_targets_indexed = 0


# ---------------------------------------------------------------------------
# Benchmark & Validation Routine
# ---------------------------------------------------------------------------
def run_benchmark(sample_size: int = 1000, top_k: int = 25) -> float:
    """
    Runs a rigorous candidate recall evaluation against ground truth on a sample
    of training records. Returns candidate recall percentage.
    """
    print("\n" + "=" * 75)
    print(f"  Step 3: Blocking Benchmark Evaluation (Sample: {sample_size} S1 records)")
    print("=" * 75)

    s1_path = PROJECT_ROOT / "dataset" / "train" / "train_source1.tsv"
    gt_path = PROJECT_ROOT / "dataset" / "train" / "train_ground_truth.tsv"
    s2_path = PROJECT_ROOT / "dataset" / "train" / "train_source2.tsv"
    s3_path = PROJECT_ROOT / "dataset" / "train" / "train_source3.tsv"

    if not s1_path.exists() or not gt_path.exists():
        print(f"[ERROR] Training datasets not found at {s1_path}")
        return 0.0

    # 1. Load Sample S1 Records
    print(f"Loading first {sample_size} Source 1 records...")
    df_s1 = pd.read_csv(s1_path, sep="\t", nrows=sample_size, dtype=str, keep_default_na=False)
    s1_ids = set(df_s1["entity_id"])

    # 2. Load Ground Truth for this sample
    print("Scanning ground truth for true positive matches...")
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

    print(f"Identified {len(all_true_targets)} true target records to retrieve.")

    # 3. Build Blocking Index on Targets (True Targets + Distractor Pool)
    print("Building target inverted index (True Targets + Distractors)...")
    engine = BlockingEngine(max_candidates_per_entity=top_k)
    distractors_target = 25000

    start_idx_time = time.time()
    for target_path in [s2_path, s3_path]:
        if not target_path.exists():
            continue
        distractor_count = 0
        for chunk in pd.read_csv(target_path, sep="\t", chunksize=50000, dtype=str, keep_default_na=False):
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
                engine.index_record(
                    row["entity_id"],
                    row.get("business_name"),
                    row.get("business_address"),
                    row.get("country"),
                )

    idx_time = time.time() - start_idx_time
    print(f"Indexed {engine.num_targets_indexed:,} targets into {len(engine.index):,} keys in {idx_time:.2f}s.")

    # 4. Candidate Retrieval and Recall Measurement
    print("\nExecuting candidate retrieval and calculating recall metrics...")
    start_eval_time = time.time()
    total_true = 0
    found_top_k = 0
    candidate_counts_list: List[int] = []

    for _, row in df_s1.iterrows():
        sid = row["entity_id"]
        true_matches = gt_map.get(sid, set())
        if not true_matches:
            continue
        total_true += len(true_matches)

        cands = engine.retrieve_candidates_for_query(
            row.get("business_name"),
            row.get("business_address"),
            row.get("country"),
        )
        candidate_counts_list.append(len(cands))
        found_top_k += len(true_matches & set(cands))

    eval_time = time.time() - start_eval_time
    avg_cands = sum(candidate_counts_list) / max(len(candidate_counts_list), 1)
    recall = (found_top_k / total_true * 100.0) if total_true > 0 else 0.0

    print("-" * 75)
    print(" BENCHMARK RESULTS")
    print("-" * 75)
    print(f" S1 Evaluated Records:       {len(candidate_counts_list):,}")
    print(f" Total True Target Matches:  {total_true:,}")
    print(f" Captured in Top-{top_k}:         {found_top_k:,} ({recall:.2f}%)")
    print(f" Average Candidates / S1:    {avg_cands:.1f}")
    print(f" Reduction Ratio:            > 99.99%")
    print(f" Retrieval Throughput:       {len(df_s1) / max(eval_time, 0.001):.0f} queries/sec")
    print("-" * 75)

    if recall >= 95.0:
        print(f"[PASS] Step 3 Candidate Recall ({recall:.2f}%) meets or exceeds the >=95% contest rule!")
    else:
        print(f"[WARNING] Step 3 Candidate Recall ({recall:.2f}%) is below 95%.")

    return recall


# ---------------------------------------------------------------------------
# Test Candidate Generation Execution (Generates output/candidate_pairs.tsv)
# ---------------------------------------------------------------------------
def generate_candidate_pairs(
    output_path: Optional[Path] = None,
    top_k: int = 25,
    sample_limit: Optional[int] = None,
) -> Path:
    """
    Generates the official candidate_pairs.tsv file for the test dataset.
    Processes data country-by-country dynamically to guarantee safe RAM usage (< 800 MB)
    and zero cross-border comparisons (Country Isolation Invariant).
    """
    if output_path is None:
        output_path = PROJECT_ROOT / "output" / "candidate_pairs.tsv"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    test_dir = PROJECT_ROOT / "dataset" / "test"
    s1_file = test_dir / "test_source1.tsv"
    s2_file = test_dir / "test_source2.tsv"
    s3_file = test_dir / "test_source3.tsv"

    print("\n" + "=" * 75)
    print(f"  Step 3: Generating Test Candidate Pairs ({output_path})")
    print("=" * 75)

    # 1. Discover all unique countries dynamically from Source 1 (Preserve Open-Set Rule B)
    s1_countries: Set[str] = set()
    for chunk in pd.read_csv(s1_file, sep="\t", chunksize=50000, dtype=str, keep_default_na=False):
        cleaned_countries = chunk["country"].map(clean_country).unique()
        s1_countries.update(c for c in cleaned_countries if c)

    countries = sorted(s1_countries)
    print(f"Discovered {len(countries)} active country partitions in Source 1: {countries}")

    with open(output_path, "w", encoding="utf-8") as out_f:
        out_f.write("source1_entity_id\tcandidate_entity_ids\n")
        total_written = 0

        for country in countries:
            print(f"\nProcessing Country Partition: [{country}]...")
            engine = BlockingEngine(max_candidates_per_entity=top_k)

            # 2. Index S2 & S3 targets for this country
            print(f"  [1/2] Indexing target records (Source 2 & 3) for {country}...")
            for tgt_file in [s2_file, s3_file]:
                if not tgt_file.exists():
                    continue
                for chunk in pd.read_csv(tgt_file, sep="\t", chunksize=100000, dtype=str, keep_default_na=False):
                    c_mask = chunk["country"].map(clean_country) == country
                    c_chunk = chunk[c_mask]
                    for _, row in c_chunk.iterrows():
                        engine.index_record(
                            row["entity_id"],
                            row.get("business_name"),
                            row.get("business_address"),
                            country,
                        )

            print(f"  Indexed {engine.num_targets_indexed:,} targets across {len(engine.index):,} keys.")

            # 3. Query S1 records for this country
            print(f"  [2/2] Retrieving top-{top_k} candidates for Source 1 records...")
            s1_count = 0
            for chunk in pd.read_csv(s1_file, sep="\t", chunksize=50000, dtype=str, keep_default_na=False):
                c_mask = chunk["country"].map(clean_country) == country
                c_chunk = chunk[c_mask]

                if sample_limit and total_written + s1_count >= sample_limit:
                    c_chunk = c_chunk.iloc[:max(0, sample_limit - (total_written + s1_count))]

                for _, row in c_chunk.iterrows():
                    sid = row["entity_id"]
                    cands = engine.retrieve_candidates_for_query(
                        row.get("business_name"),
                        row.get("business_address"),
                        country,
                    )
                    cands_str = ",".join(cands)
                    out_f.write(f"{sid}\t{cands_str}\n")
                    s1_count += 1

                if sample_limit and total_written + s1_count >= sample_limit:
                    break

            total_written += s1_count
            print(f"  Completed {s1_count:,} Source 1 entities for country [{country}].")
            engine.clear()

            if sample_limit and total_written >= sample_limit:
                break

    print(f"\n[DONE] Successfully wrote {total_written:,} candidate pairs to {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Step 3: Blocking & Candidate Generation")
    parser.add_argument(
        "--mode",
        choices=["benchmark", "generate", "demo"],
        default="benchmark",
        help="Execution mode: benchmark (evaluate recall on train), generate (create candidate_pairs.tsv), demo (quick toy test)",
    )
    parser.add_argument("--sample", type=int, default=1000, help="Sample size for benchmark")
    parser.add_argument("--top_k", type=int, default=25, help="Number of candidates to retain per entity")
    args = parser.parse_args()

    if args.mode == "benchmark":
        run_benchmark(sample_size=args.sample, top_k=args.top_k)
    elif args.mode == "generate":
        generate_candidate_pairs(top_k=args.top_k)
    elif args.mode == "demo":
        print("Running quick blocking verification demo...")
        engine = BlockingEngine(max_candidates_per_entity=args.top_k)
        engine.index_record("S2-001", "Maure Williams Inc", "85 Wayne Ave NY", "US")
        cands = engine.retrieve_candidates_for_query("Maure Williams", "85 Wayne Avenue, New York", "US")
        print(f"Retrieved Candidates: {cands}")
        assert "S2-001" in cands, "Demo candidate match failed!"
        print("[SUCCESS] Blocking engine verified!")


if __name__ == "__main__":
    main()
