# Amazon ML Challenge - Business Entity Resolution

## Project Purpose
The goal of this challenge is to perform **Business Entity Resolution**: matching Source 1 business records with corresponding records from Source 2 and Source 3 using only the provided datasets.

---

## Folder Structure

```text
Amazon ML/
│
├── dataset/
│   ├── train/
│   │   ├── train_source1.tsv
│   │   ├── train_source2.tsv
│   │   ├── train_source3.tsv
│   │   └── train_ground_truth.tsv
│   │
│   └── test/
│       ├── test_source1.tsv
│       ├── test_source2.tsv
│       └── test_source3.tsv
│
├── src/
│   ├── __init__.py
│   ├── data_inspection.py
│   ├── data_cleaning.py
│   └── blocking.py
│
├── utils/
│   └── validate_submission.py
│
├── output/
├── Documentation_template.md
├── requirements.txt
└── README.md
```

- **`dataset/train/`**: Training TSV files (`train_source1.tsv`, `train_source2.tsv`, `train_source3.tsv`) and labels (`train_ground_truth.tsv`).
- **`dataset/test/`**: Test TSV files (`test_source1.tsv`, `test_source2.tsv`, `test_source3.tsv`).
- **`src/`**: Source code modules:
  - `src/__init__.py`: Package marker.
  - `src/data_inspection.py`: Inspects dataset shape, columns, missing values, duplicates, and preview rows.
  - `src/data_cleaning.py`: Normalizes noisy business names and addresses (stripping legal suffixes, standardizing abbreviations across US, India, and France).
  - `src/blocking.py`: High-recall multi-key candidate generation and token Jaccard candidate ranking.
- **`utils/validate_submission.py`**: Official submission validation script to verify format compliance before leaderboard submission.
- **`output/`**: Directory reserved for intermediate candidates (`candidate_pairs.tsv`) and final predictions (`matching_results.tsv`).
- **`Documentation_template.md`**: Template for contest methodology write-up.
- **`requirements.txt`**: Python dependencies required for the project.

---

## Setup Instructions

### 1. Install Dependencies
Open your terminal inside the project root (`Amazon ML`) and install dependencies:

```bash
pip install -r requirements.txt
```

---

## How to Run Modules

### 1. Data Inspection (Step 1)
```bash
python src/data_inspection.py
```

### 2. Data Cleaning & Normalization (Step 2)
```bash
python src/data_cleaning.py
```
This tests and demonstrates the normalization routines on representative business names and addresses across US, India, and France.

### 3. Blocking & Candidate Generation (Step 3)
```bash
python src/blocking.py
```
This demonstrates the multi-key inverted index blocking mechanism and lightweight candidate retrieval.
