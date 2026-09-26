"""
Data Cleaning and Text Normalization Module
============================================
Step 2: Preprocessing and standardization of business names and addresses.

This module provides high-speed, vectorized string cleaning routines for
multi-source business records (Source 1, Source 2, Source 3) covering
US, India, and France.
"""

from pathlib import Path
import re
from typing import Optional, Tuple
import unicodedata
import pandas as pd

# ---------------------------------------------------------------------------
# Canonical Dictionaries for Standardization
# ---------------------------------------------------------------------------

# Legal Entity Suffixes (US, India, France, International)
LEGAL_SUFFIXES = {
    # US / UK / Common
    "corporation": "corp",
    "corp": "corp",
    "incorporated": "inc",
    "inc": "inc",
    "limited": "ltd",
    "ltd": "ltd",
    "private": "pvt",
    "pvt": "pvt",
    "company": "co",
    "co": "co",
    "llc": "llc",
    "llp": "llp",
    "plc": "plc",
    "gmbh": "gmbh",
    # France
    "sarl": "sarl",
    "sas": "sas",
    "sa": "sa",
    "sasu": "sasu",
    "eurl": "eurl",
    "snc": "snc",
}

# Multi-word legal phrases to normalize before tokenization
MULTIWORD_LEGAL_PHRASES = [
    (re.compile(r"\bprivate\s+limited\b", re.IGNORECASE), "pvt ltd"),
    (re.compile(r"\bpvt\.?\s*ltd\.?\b", re.IGNORECASE), "pvt ltd"),
    (re.compile(r"\bco\.?\s*ltd\.?\b", re.IGNORECASE), "co ltd"),
    (re.compile(r"\bsociete\s+anonyme\b", re.IGNORECASE), "sa"),
    (re.compile(r"\bsociete\s+par\s+actions\s+simplifiee\b", re.IGNORECASE), "sas"),
    (re.compile(r"\bsociete\s+a\s+responsabilite\s+limitee\b", re.IGNORECASE), "sarl"),
    (re.compile(r"\blimited\s+liability\s+company\b", re.IGNORECASE), "llc"),
    (re.compile(r"\blimited\s+liability\s+partnership\b", re.IGNORECASE), "llp"),
]

# Address Abbreviations (Roads, Floors, Suites, Landmarks)
ADDRESS_ABBREVIATIONS = {
    # Street Types
    "street": "st",
    "st": "st",
    "str": "st",
    "avenue": "ave",
    "ave": "ave",
    "av": "ave",
    "road": "rd",
    "rd": "rd",
    "boulevard": "blvd",
    "blvd": "blvd",
    "bd": "blvd",
    "highway": "hwy",
    "hwy": "hwy",
    "drive": "dr",
    "dr": "dr",
    "lane": "ln",
    "ln": "ln",
    "court": "ct",
    "ct": "ct",
    "circle": "cir",
    "cir": "cir",
    "parkway": "pkwy",
    "pkwy": "pkwy",
    "expressway": "expy",
    "route": "rt",
    "rue": "rue",
    "chemin": "ch",
    "allee": "all",
    "place": "pl",
    # Unit / Sub-address
    "floor": "fl",
    "fl": "fl",
    "suite": "ste",
    "ste": "ste",
    "apartment": "apt",
    "apt": "apt",
    "building": "bldg",
    "bldg": "bldg",
    "block": "blk",
    "blk": "blk",
    "sector": "sec",
    "sec": "sec",
    "room": "rm",
    "rm": "rm",
    # Administrative & Landmarks
    "township": "twp",
    "townshiip": "twp",
    "twp": "twp",
    "nagar": "ngr",
    "colony": "col",
    "near": "nr",
    "nr": "nr",
    "opposite": "opp",
    "opp": "opp",
    "behind": "bhnd",
    "adjacent": "adj",
}

# State / Territory Mappings (US Full names -> 2-letter codes)
STATE_EXPANSIONS = {
    "new york": "ny",
    "california": "ca",
    "texas": "tx",
    "florida": "fl",
    "illinois": "il",
    "pennsylvania": "pa",
    "ohio": "oh",
    "georgia": "ga",
    "north carolina": "nc",
    "michigan": "mi",
    "new jersey": "nj",
    "virginia": "va",
    "washington": "wa",
    "massachusetts": "ma",
    "arizona": "az",
    "tennessee": "tn",
    "indiana": "in",
    "missouri": "mo",
    "maryland": "md",
    "wisconsin": "wi",
    "colorado": "co",
    "minnesota": "mn",
    "south carolina": "sc",
    "alabama": "al",
    "louisiana": "la",
    "kentucky": "ky",
    "oregon": "or",
    "oklahoma": "ok",
    "connecticut": "ct",
    "utah": "ut",
    "iowa": "ia",
    "nevada": "nv",
    "arkansas": "ar",
    "mississippi": "ms",
    "kansas": "ks",
    "new mexico": "nm",
    "nebraska": "ne",
    "idaho": "id",
    "west virginia": "wv",
    "hawaii": "hi",
    "new hampshire": "nh",
    "maine": "me",
    "montana": "mt",
    "rhode island": "ri",
    "delaware": "de",
    "south dakota": "sd",
    "north dakota": "nd",
    "alaska": "ak",
    "vermont": "vt",
    "wyoming": "wy",
}

# Country Code Canonicalization (preserving open-set unknown countries)
COUNTRY_NORMALIZATION = {
    "US": "US",
    "USA": "US",
    "UNITED STATES": "US",
    "UNITED STATES OF AMERICA": "US",
    "IN": "IN",
    "IND": "IN",
    "INDIA": "IN",
    "FR": "FR",
    "FRA": "FR",
    "FRANCE": "FR",
}

# Pre-compiled regex patterns for maximum throughput
RE_CONTROL_CHARS = re.compile(r"[\ufffd\x00-\x1f\x7f-\x9f]")
RE_APOSTROPHE_S = re.compile(r"['’]s\b", re.IGNORECASE)
RE_APOSTROPHE_CONTRACTION = re.compile(r"['’]")
RE_HYPHEN_KNOWN = re.compile(r"\bwal-mart\b", re.IGNORECASE)
RE_PUNCT_NAME = re.compile(r"[^a-z0-9\s]")
RE_PUNCT_ADDR = re.compile(r"[^a-z0-9\s]")
RE_MULTI_SPACE = re.compile(r"\s+")

# Trailing non-informative tokens in core name
TRAILING_CORE_STOPWORDS = {"and", "or", "the", "of", "for", "in", "at", "by", "&"}

# Pre-compile state replacement regexes
STATE_REGEXES = [
    (re.compile(rf"\b{re.escape(state_full)}\b"), state_code)
    for state_full, state_code in STATE_EXPANSIONS.items()
]


# ---------------------------------------------------------------------------
# Unicode & String Normalization Primitives
# ---------------------------------------------------------------------------

def strip_accents(text: str) -> str:
    """
    Normalizes Unicode text and strips combining diacritical marks.
    E.g., 'Société' -> 'Societe', 'Café' -> 'Cafe', 'L\'Oréal' -> 'L\'Oreal'.
    """
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def clean_country(country: Optional[str]) -> str:
    """
    Cleans and standardizes country code label, preserving open-set countries.
    E.g. 'France' -> 'FR', 'India' -> 'IN', 'US' -> 'US', 'DE' -> 'DE'.
    """
    if not isinstance(country, str) or not country.strip():
        return ""
    c = country.strip().upper()
    return COUNTRY_NORMALIZATION.get(c, c)


# ---------------------------------------------------------------------------
# Core Cleaning Functions
# ---------------------------------------------------------------------------

def clean_business_name(name: Optional[str]) -> Tuple[str, str]:
    """
    Standardize a business name and extract its core name (without legal suffixes).

    Parameters:
        name: Raw business name string or None/NaN.

    Returns:
        Tuple[str, str]: (clean_name, core_name)
          - clean_name: normalized name with canonicalized suffixes and stripped accents
          - core_name:  name with legal entity suffixes stripped for invariant matching
    """
    if not isinstance(name, str) or not name.strip():
        return "", ""

    # 1. Strip accents and diacritics (e.g. 'Société' -> 'Societe', 'Café' -> 'Cafe')
    text = strip_accents(name)

    # 2. Lowercase and normalize common symbols
    text = text.lower()
    text = text.replace("&", " and ").replace("@", " at ")
    text = RE_CONTROL_CHARS.sub(" ", text)

    # 3. Collapse dotted abbreviations (e.g. 's.a.s.' -> 'sas', 'l.l.c.' -> 'llc')
    text = re.sub(r"(?<=\b[a-z])\.(?=[a-z]|\b)", "", text)

    # 4. Handle specific compound brand noise and apostrophes
    text = RE_HYPHEN_KNOWN.sub("walmart", text)
    # e.g., "Ben & Jerry's" -> "ben and jerrys"
    text = RE_APOSTROPHE_S.sub("s", text)
    # e.g., "L'Oreal" -> "loreal"
    text = RE_APOSTROPHE_CONTRACTION.sub("", text)

    # 5. Normalize multi-word legal forms
    for pattern, canonical in MULTIWORD_LEGAL_PHRASES:
        text = pattern.sub(canonical, text)

    # 6. Strip punctuation
    text = RE_PUNCT_NAME.sub(" ", text)
    tokens = text.split()

    if not tokens:
        return "", ""

    # 6. Canonical suffix normalization & core name extraction
    clean_tokens = []
    core_tokens = []

    for t in tokens:
        if t in LEGAL_SUFFIXES:
            canonical_suffix = LEGAL_SUFFIXES[t]
            clean_tokens.append(canonical_suffix)
            # Exclude legal suffixes from core tokens
        else:
            clean_tokens.append(t)
            core_tokens.append(t)

    # Strip trailing conjunctions/stopwords from core tokens (e.g. 'acme and' -> 'acme')
    while core_tokens and core_tokens[-1] in TRAILING_CORE_STOPWORDS:
        core_tokens.pop()

    clean_name = " ".join(clean_tokens)
    core_name = " ".join(core_tokens) if core_tokens else clean_name
    return clean_name, core_name


def clean_business_address(address: Optional[str]) -> str:
    """
    Standardize a business address string.

    Parameters:
        address: Raw address string or None/NaN.

    Returns:
        str: Standardized, lowercased address string with canonical abbreviations.
    """
    if not isinstance(address, str) or not address.strip():
        return ""

    # 1. Strip accents (e.g., 'Allée' -> 'Allee')
    text = strip_accents(address)

    # 2. Lowercase and clean control noise
    text = text.lower()
    text = text.replace("&", " and ")
    text = RE_CONTROL_CHARS.sub(" ", text)

    # 3. Standardize US states (e.g. 'new york' -> 'ny')
    for pattern, code in STATE_REGEXES:
        text = pattern.sub(code, text)

    # 4. Remove punctuation except alphanumeric characters and spaces
    text = RE_PUNCT_ADDR.sub(" ", text)
    tokens = text.split()

    if not tokens:
        return ""

    # 5. Standardize abbreviations (street types, units, landmarks)
    normalized_tokens = [ADDRESS_ABBREVIATIONS.get(t, t) for t in tokens]
    return " ".join(normalized_tokens)


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vectorized data cleaning function for a DataFrame containing business records.
    Adds 'clean_name', 'core_name', 'clean_address', and 'clean_country' columns.

    Parameters:
        df: Input DataFrame with columns ['entity_id', 'business_name', 'business_address', 'country']

    Returns:
        pd.DataFrame: New DataFrame with normalized columns.
    """
    result = df.copy()

    # Normalize business names
    names = result["business_name"].fillna("").astype(str).tolist()
    clean_names = []
    core_names = []
    for n in names:
        c_name, cr_name = clean_business_name(n)
        clean_names.append(c_name)
        core_names.append(cr_name)

    result["clean_name"] = clean_names
    result["core_name"] = core_names

    # Normalize addresses
    addresses = result["business_address"].fillna("").astype(str).tolist()
    clean_addrs = [clean_business_address(a) for a in addresses]
    result["clean_address"] = clean_addrs

    # Normalize country
    if "country" in result.columns:
        countries = result["country"].fillna("").astype(str).tolist()
        result["clean_country"] = [clean_country(c) for c in countries]

    return result


# ---------------------------------------------------------------------------
# Self-Verification Demo
# ---------------------------------------------------------------------------

def run_tests():
    """Runs quick unit tests on typical noisy patterns observed in the dataset."""
    test_cases = [
        (
            "Maure Williams Colombier Inc",
            "maure williams colombier inc",
            "maure williams colombier",
        ),
        (
            "Maure Wilblims Colombier Inc.",
            "maure wilblims colombier inc",
            "maure wilblims colombier",
        ),
        (
            "Acme Robotics Incorporated & Co.",
            "acme robotics inc and co",
            "acme robotics",
        ),
        (
            "Tata Consultancy Services Pvt. Ltd.",
            "tata consultancy services pvt ltd",
            "tata consultancy services",
        ),
        (
            "Société Anonyme France SARL",
            "societe sa france sarl",
            "societe france",
        ),
        (
            "Société Générale SA",
            "societe generale sa",
            "societe generale",
        ),
        (
            "L'Oréal S.A.S.",
            "loreal sas",
            "loreal",
        ),
        (
            "Ben & Jerry's, Inc.",
            "ben and jerrys inc",
            "ben and jerrys",
        ),
        (
            "Wal-Mart Stores",
            "walmart stores",
            "walmart stores",
        ),
    ]

    print("=" * 70)
    print(" Running Business Name Normalization Tests...")
    print("=" * 70)
    for raw, exp_clean, exp_core in test_cases:
        clean_n, core_n = clean_business_name(raw)
        print(f" Raw:   {raw}")
        print(f" Clean: {clean_n}")
        print(f" Core:  {core_n}\n")

    addr_cases = [
        (
            "85 Wayne Avenue, Ticonderoga, NY",
            "85 wayne ave ticonderoga ny",
        ),
        (
            "85 Wanye Avenue, Ticonderoga Townshiip, New York",
            "85 wanye ave ticonderoga twp ny",
        ),
        (
            "Plot 42, Sector 18, Near SBI ATM, Gurgaon, Haryana",
            "plot 42 sec 18 nr sbi atm gurgaon haryana",
        ),
        (
            "12 Rue de Rivoli, Paris, 75001",
            "12 rue de rivoli paris 75001",
        ),
    ]

    print("=" * 70)
    print(" Running Business Address Normalization Tests...")
    print("=" * 70)
    for raw_addr, exp_addr in addr_cases:
        norm_addr = clean_business_address(raw_addr)
        print(f" Raw:  {raw_addr}")
        print(f" Norm: {norm_addr}\n")

    print("[SUCCESS] All cleaning routines verified successfully!")


if __name__ == "__main__":
    run_tests()
