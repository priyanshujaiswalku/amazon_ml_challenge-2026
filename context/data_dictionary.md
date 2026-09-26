# Data Dictionary & Noise Primitives: Amazon ML Challenge 2026

## 1. Input Dataset Schemas

### A. Source Entity Files (`*_source1.tsv`, `*_source2.tsv`, `*_source3.tsv`)
All source files share an identical 4-column schema separated by a single tab character (`\t`):

| Column Name | Data Type | Nullable? | Prefix / Values | Description & Semantic Constraints |
|---|---|---|---|---|
| `entity_id` | String | No | `S1-` / `S2-` / `S3-` | Unique primary key identifying the record within its source file. |
| `business_name` | String | Yes (Rare) | Free-text | Name of the business entity; contains abbreviations, typos, legal suffixes. |
| `business_address`| String | Yes (Rare) | Free-text | Physical location text; contains abbreviations, landmarks, postal codes. |
| `country` | String | No | `US`, `IN`, `FR` | Geographic sovereignty. Training has `US`, `IN`. Test adds `FR`. |

### B. Ground Truth File (`train/train_ground_truth.tsv`)
Defines the true matches for each training Source 1 entity:

| Column Name | Data Type | Nullable? | Format / Values | Description & Rules |
|---|---|---|---|---|
| `source1_entity_id`| String | No | `S1-xxxxx` | Every training $S_1$ entity ID appears exactly once. |
| `matched_entity_ids`| String | Yes | Comma-separated | IDs of matching entities from $S_2$ and/or $S_3$. Empty string `""` for singletons. |

---

## 2. Output Dataset Schemas

### A. Final Leaderboard Matches (`output/matching_results.tsv`)
The only file evaluated on the competition leaderboard.

| Column Name | Format Example | Strict Constraints |
|---|---|---|
| `source1_entity_id` | `S1-00001` | Must contain every $S_1$ entity from `test_source1.tsv` exactly once. |
| `matched_entity_ids` | `S2-00047,S3-00812` | Comma-separated list of target IDs. Empty string for singletons. No quotes. No $S_1$ IDs. No duplicate IDs. |

### B. Blocking Candidate Set (`output/candidate_pairs.tsv`)
The candidate set output by the blocking stage just before machine learning scoring.

| Column Name | Format Example | Strict Constraints |
|---|---|---|
| `source1_entity_id` | `S1-00001` | Must match all test $S_1$ entities. |
| `candidate_entity_ids` | `S2-00047,S2-00193,S3-00812` | Comma-separated list of candidate target IDs. Must be a superset of `matching_results.tsv`. |

---

## 3. Geographic Data Primitives & Country Profiles

### A. United States (`country == 'US'`)
- **Address Structure**: Highly structured: `[Street Number] [Street Name] [Street Suffix] [Unit/Suite] [City] [State 2-letter Code] [5-digit ZIP]`.
- **Key Abbreviations**: `St` (Street), `Ave` (Avenue), `Blvd` (Boulevard), `Hwy` (Highway), `Ste` (Suite), `Fl` (Floor).
- **Legal Suffixes**: `Inc`, `Corp`, `LLC`, `LLP`, `Co`, `Ltd`, `Company`.
- **Postal Code**: 5-digit ZIP code (`^\d{5}$` or `^\d{5}-\d{4}$`).

### B. India (`country == 'IN'`)
- **Address Structure**: Semi-structured and landmark-oriented: `[Door/Plot No.] [Building/Society] [Landmark] [Sector/Colony/Nagar] [City] [6-digit PIN]`.
- **Key Landmarks**: `Near`, `Opposite` (`Opp`), `Behind`, `Beside`, `Adjacent`, `Above`, `Floor`.
- **Locality Tokens**: `Nagar`, `Colony`, `Enclave`, `Vihar`, `Bagh`, `Gali`, `Marg`, `Chowk`, `Bazaar`.
- **Legal Suffixes**: `Pvt Ltd`, `Private Limited`, `LLP`, `Ltd`, `Enterprises`, `Brothers`, `Stores`.
- **Postal Code**: 6-digit PIN code (`^[1-9]\d{5}$`).
- **Phonetic Variations**: High prevalence of Hindi/regional phonetic spellings (e.g., `Laxmi` vs `Lakshmi`, `Khandelwal` vs `Khandelval`).

### C. France (`country == 'FR'`) — Zero-Shot Test Evaluation
- **Address Structure**: `[Number] [Street Type] [Street Name], [5-digit Code Postal] [Commune/City]`.
- **Street Types**: `Rue` (Street), `Avenue` (`Ave`), `Boulevard` (`Bd`/`Blvd`), `Chemin` (`Ch`), `Allée` (`All`), `Place` (`Pl`), `Route` (`Rt`).
- **Legal Suffixes**: `SARL` (Société à responsabilité limitée), `SAS` (Société par actions simplifiée), `SA` (Société anonyme), `SASU`, `EURL`, `SNC`.
- **Postal Code**: 5-digit postal code (`^\d{5}$`).
- **Character Encoding**: French accents (`é`, `è`, `ê`, `ë`, `à`, `â`, `ï`, `ô`, `ç`). Must normalize or preserve accents consistently across sources.

---

## 4. Noise Taxonomy & Standardization Mapping

```text
Raw Noise Type          Input Variation Example                 Canonical Target
──────────────────────────────────────────────────────────────────────────────────
Legal Suffixes          "Acme Corporation Inc."                 "acme"
                        "Reliance Industries Pvt. Ltd."         "reliance industries"
                        "L'Oreal SA"                            "loreal"

Street Types            "100 Main Street"                       "100 main st"
                        "12 Boulevard Haussmann"                "12 blvd haussmann"
                        "MG Road"                               "mg rd"

Indian Landmarks        "Opposite SBI Bank, Near Bus Stand"     "opp sbi bank nr bus stand"
                        "Behind Apollo Pharmacy"                "behind apollo pharmacy"

Punctuation & Typos     "Ben & Jerry's, Inc."                   "ben and jerrys"
                        "Wal-Mart Stores"                       "walmart"
                        "Mc Donald's"                           "mcdonalds"

Special Characters      "Café de la Gare"                       "cafe de la gare"
                        "Société Générale"                      "societe generale"
```
