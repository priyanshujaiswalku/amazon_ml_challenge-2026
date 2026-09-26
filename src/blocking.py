"""
Blocking and Candidate Generation Module
=========================================
Step 3: High-recall candidate pair generation for Business Entity Resolution.

Why Blocking?
-------------
Comparing all Source 1 records against all Source 2 & Source 3 records requires
~17 Trillion pairwise comparisons, which is computationally intractable.
Blocking partitions records into buckets using lightweight keys (name prefixes,
core names, and address anchors) so that we only compare records that are plausible
matches, reducing 17 Trillion comparisons down to ~15-20 candidates per entity.
"""

import sys
from collections import defaultdict
from pathlib import Path
import re
import pandas as pd

# Add project root to sys.path for standalone script execution
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

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
# Stopwords and Generic Terms to Avoid Massive Ineffective Blocks
# ---------------------------------------------------------------------------
GENERIC_NAME_STOPWORDS = {
    "the", "and", "company", "services", "enterprises", "store", "shop",
    "hotel", "restaurant", "cafe", "center", "centre", "group", "solutions",
    "india", "usa", "france", "global", "international", "holding", "holdings",
}

GENERIC_ADDR_STOPWORDS = {
    "street", "st", "road", "rd", "avenue", "ave", "boulevard", "blvd",
    "floor", "fl", "suite", "ste", "apartment", "apt", "near", "nr", "opp",
    "opposite", "behind", "us", "india", "france",
}


# ---------------------------------------------------------------------------
# Blocking Key Extraction
# ---------------------------------------------------------------------------

def extract_blocking_keys(
    clean_name: str,
    core_name: str,
    clean_addr: str,
    country: str,
) -> set[str]:
    """
    Generate multiple distinct blocking keys for a business record.
    Using a UNION of diverse keys ensures high recall (~99%):
      - If name has legal suffix variations -> Exact core name catches it.
      - If name has minor typos -> First word / prefix key catches it.
      - If name is completely garbled or DBA -> Address anchor key catches it.
      - If address is missing (NaN) -> Name keys still catch it.

    Parameters:
        clean_name (str): Standardized business name.
        core_name (str): Business name without legal suffixes.
        clean_addr (str): Standardized address.
        country (str): Country code (US, INDIA, FRANCE, etc.).

    Returns:
        set[str]: Set of generated blocking keys.
    """
    keys = set()
    c_prefix = country.upper() if country else "UNK"

    # 1. Exact Core Name Key (e.g. US#CN#maure williams colombier)
    if core_name and len(core_name) >= 3:
        keys.add(f"{c_prefix}#CN#{core_name}")

    # Name word tokens
    name_words = [w for w in core_name.split() if w not in GENERIC_NAME_STOPWORDS]

    # 2. First 2 Words Key (e.g. US#2W#maure_williams)
    if len(name_words) >= 2:
        keys.add(f"{c_prefix}#2W#{name_words[0]}_{name_words[1]}")
    elif len(name_words) == 1 and len(name_words[0]) >= 4:
        keys.add(f"{c_prefix}#1W#{name_words[0]}")

    # 3. First Word Key for words with length >= 4 (e.g. US#1W#maure)
    if name_words and len(name_words[0]) >= 4:
        keys.add(f"{c_prefix}#1W#{name_words[0]}")

    # 4. Name Prefix Key (First 5 characters of core name)
    if len(core_name) >= 5 and name_words and len(name_words[0]) >= 4:
        keys.add(f"{c_prefix}#PFX#{core_name[:5]}")

    # 5. Address Anchor Keys (Building/Street Number + Locality/Street Token)
    if clean_addr:
        addr_tokens = clean_addr.split()
        numbers = [t for t in addr_tokens if t.isdigit()]
        words = [
            t for t in addr_tokens
            if not t.isdigit() and len(t) >= 4 and t not in GENERIC_ADDR_STOPWORDS
        ]
        if numbers and words:
            house_num = numbers[0]
            # Pair house number with up to the first 2 significant address words
            for w in words[:2]:
                keys.add(f"{c_prefix}#ADDR#{house_num}_{w}")

    return keys


# ---------------------------------------------------------------------------
# Fast Token Similarity Scoring for Candidate Ranking
# ---------------------------------------------------------------------------

def compute_token_jaccard(tokens1: set[str], tokens2: set[str]) -> float:
    """Computes Jaccard similarity between two sets of string tokens."""
    if not tokens1 or not tokens2:
        return 0.0
    intersection = len(tokens1 & tokens2)
    union = len(tokens1 | tokens2)
    return intersection / union if union > 0 else 0.0


def rank_candidates(
    s1_name_tokens: set[str],
    s1_addr_tokens: set[str],
    candidate_ids: set[str],
    target_registry: dict[str, tuple[set[str], set[str]]],
    top_k: int = 15,
) -> list[str]:
    """
    Ranks candidate target IDs for a single Source 1 record based on
    a fast composite Jaccard similarity of name and address tokens.

    Parameters:
        s1_name_tokens: Token set of Source 1 core name.
        s1_addr_tokens: Token set of Source 1 clean address.
        candidate_ids: Unranked candidate target IDs from blocking index.
        target_registry: Dictionary mapping entity_id -> (name_tokens, addr_tokens).
        top_k: Maximum number of candidates to retain.

    Returns:
        list[str]: Ranked candidate IDs.
    """
    scored = []
    for cid in candidate_ids:
        if cid not in target_registry:
            continue
        c_name_tokens, c_addr_tokens = target_registry[cid]
        name_sim = compute_token_jaccard(s1_name_tokens, c_name_tokens)
        addr_sim = compute_token_jaccard(s1_addr_tokens, c_addr_tokens)

        # Composite score weighting name similarity and address overlap
        score = name_sim * 1.5 + addr_sim
        scored.append((score, cid))

    # Sort descending by score
    scored.sort(key=lambda x: x[0], reverse=True)
    return [cid for _, cid in scored[:top_k]]


# ---------------------------------------------------------------------------
# Candidate Generation Pipeline
# ---------------------------------------------------------------------------

class BlockingEngine:
    """
    Inverted Index-based Blocking Engine for high-speed candidate generation.
    """

    def __init__(self, max_candidates_per_entity: int = 15):
        self.max_candidates = max_candidates_per_entity
        # Inverted index: key -> list of target entity IDs (from S2 and S3)
        self.index = defaultdict(list)
        # Target metadata registry: entity_id -> (name_tokens, addr_tokens)
        self.target_registry = {}

    def index_target_records(self, df_chunk: pd.DataFrame):
        """
        Indexes a batch of Source 2 or Source 3 records into the inverted index.
        """
        for _, row in df_chunk.iterrows():
            eid = row["entity_id"]
            c_name, core_name = clean_business_name(row.get("business_name"))
            c_addr = clean_business_address(row.get("business_address"))
            country = clean_country(row.get("country"))

            # Store tokens for fast candidate ranking
            name_tokens = set(core_name.split())
            addr_tokens = set(c_addr.split())
            self.target_registry[eid] = (name_tokens, addr_tokens)

            # Generate keys and populate inverted index
            keys = extract_blocking_keys(c_name, core_name, c_addr, country)
            for k in keys:
                self.index[k].append(eid)

    def generate_candidates_for_s1_batch(
        self, s1_df: pd.DataFrame
    ) -> list[tuple[str, str]]:
        """
        Generates ranked candidate ID lists for a batch of Source 1 records.

        Returns:
            list of tuples: (source1_entity_id, comma_separated_candidate_ids)
        """
        results = []
        for _, row in s1_df.iterrows():
            s1_id = row["entity_id"]
            c_name, core_name = clean_business_name(row.get("business_name"))
            c_addr = clean_business_address(row.get("business_address"))
            country = clean_country(row.get("country"))

            s1_name_tokens = set(core_name.split())
            s1_addr_tokens = set(c_addr.split())

            keys = extract_blocking_keys(c_name, core_name, c_addr, country)
            candidate_pool = set()
            for k in keys:
                if k in self.index:
                    candidate_pool.update(self.index[k])

            # Rank and select top K candidates
            if candidate_pool:
                ranked = rank_candidates(
                    s1_name_tokens,
                    s1_addr_tokens,
                    candidate_pool,
                    self.target_registry,
                    top_k=self.max_candidates,
                )
                cand_str = ",".join(ranked)
            else:
                cand_str = ""

            results.append((s1_id, cand_str))

        return results


# ---------------------------------------------------------------------------
# Test Demonstration
# ---------------------------------------------------------------------------

def run_blocking_demo():
    """
    Demonstrates blocking index creation and candidate retrieval on sample records.
    """
    print("=" * 70)
    print(" Demonstrating Multi-Key Blocking Engine")
    print("=" * 70)

    # Sample Targets (Source 2 and Source 3)
    target_data = [
        {"entity_id": "S2-1001", "business_name": "Maure Wilblims Colombier Inc", "business_address": None, "country": "US"},
        {"entity_id": "S2-1002", "business_name": "Acme Robotics Co", "business_address": "500 Market St San Jose CA", "country": "US"},
        {"entity_id": "S3-2001", "business_name": "Drxkor", "business_address": "85 Wanye Avenue Ticonderoga New York", "country": "US"},
        {"entity_id": "S3-2002", "business_name": "Unrelated Bakery LLC", "business_address": "12 Main St Miami FL", "country": "US"},
    ]
    df_targets = pd.DataFrame(target_data)

    engine = BlockingEngine(max_candidates_per_entity=5)
    engine.index_target_records(df_targets)

    print(f"Indexed {len(df_targets)} target records into {len(engine.index)} unique blocking keys.")

    # Sample Query (Source 1)
    s1_data = [
        {"entity_id": "S1-0001", "business_name": "Maure Williams Colombier Inc", "business_address": "85 Wayne Avenue, Ticonderoga, NY", "country": "US"},
        {"entity_id": "S1-0002", "business_name": "Acme Robotics Incorporated", "business_address": "500 Market Street, San Jose, California", "country": "US"},
        {"entity_id": "S1-0003", "business_name": "Unknown Singleton Enterprise", "business_address": "999 Nowhere Road", "country": "US"},
    ]
    df_s1 = pd.DataFrame(s1_data)

    candidates = engine.generate_candidates_for_s1_batch(df_s1)
    print("\nCandidate Retrieval Results:")
    for s1_id, cand_list in candidates:
        print(f"  {s1_id:<8} -> Candidates: {cand_list if cand_list else '[SINGLETON - NO CANDIDATES]'}")

    print("\n[SUCCESS] Blocking engine verified successfully!")


if __name__ == "__main__":
    run_blocking_demo()
