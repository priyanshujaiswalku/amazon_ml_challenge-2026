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
│   ├── blocking.py
│   └── matching_model.py
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
  - `src/matching_model.py`: Pairwise feature extraction, LightGBM classification, F_0.5 threshold optimization, and final prediction export.
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
python src/blocking.py --mode benchmark
```
Evaluates candidate recall on ground truth and demonstrates the multi-key inverted index blocking mechanism.

### 4. Matching Model & Prediction (Step 4)
```bash
# Train LightGBM classifier and find optimal F_0.5 threshold
python src/matching_model.py --mode train

# Run inference and generate submission files
python src/matching_model.py --mode predict
```
This trains the pairwise model, optimizes decision threshold for $F_{0.5}$, and exports `output/matching_results.tsv` and `output/candidate_pairs.tsv`.
