"""
Data Inspection Script for Amazon ML Challenge - Business Entity Resolution
===========================================================================
Step 1: Project setup and dataset inspection.

This script inspects each TSV dataset in the train and test directories,
displaying key statistics including shape, columns, sample rows, missing values,
and duplicate entity IDs.
"""

from pathlib import Path
import pandas as pd


def inspect_dataset(dataset_name: str, file_path: Path) -> pd.DataFrame | None:
    """
    Load and inspect a single TSV dataset file.

    Parameters:
        dataset_name (str): Friendly name for the dataset.
        file_path (Path): Pathlib Path object pointing to the TSV file.

    Returns:
        pd.DataFrame or None: The loaded DataFrame if successful, else None.
    """
    separator_line = "=" * 70
    print(f"\n{separator_line}")
    print(f" Dataset: {dataset_name}")
    print(f" File:    {file_path}")
    print(f"{separator_line}")

    # Check if file exists to avoid crashing with an unclear traceback
    if not file_path.exists():
        print(f"\n[ERROR] File not found: {file_path}")
        print("Please ensure the dataset file is placed in the directory specified above.")
        return None

    try:
        # Load the TSV file using tab delimiter as specified
        df = pd.read_csv(file_path, sep="\t")
    except Exception as e:
        print(f"\n[ERROR] Failed to read '{file_path.name}': {e}")
        return None

    # 1. Dataset Name, Rows, and Columns
    num_rows, num_cols = df.shape
    print(f"\nNumber of Rows:    {num_rows}")
    print(f"Number of Columns: {num_cols}")

    # 2. Column Names
    print("\nColumn Names:")
    for idx, col in enumerate(df.columns, start=1):
        print(f"  {idx}. {col}")

    # 3. First 5 Rows
    print("\nFirst 5 Rows:")
    print(df.head())

    # 4. Missing Values in Each Column
    print("\nMissing Values per Column:")
    missing_series = df.isnull().sum()
    for col, missing_count in missing_series.items():
        missing_pct = (missing_count / num_rows * 100) if num_rows > 0 else 0.0
        print(f"  - {col}: {missing_count} missing ({missing_pct:.2f}%)")

    # 5. Duplicate entity_id Count (where entity_id exists)
    if "entity_id" in df.columns:
        dup_count = df["entity_id"].duplicated().sum()
        print(f"\nDuplicate 'entity_id' count: {dup_count}")
    else:
        print("\n'entity_id' column is not present in this dataset.")

    return df


def main():
    """
    Main function to inspect all training and test datasets.
    """
    # Use pathlib to resolve the root folder of the project dynamically.
    # __file__ is the path to src/data_inspection.py.
    # .resolve().parent is 'src/', and .parent.parent is the 'Amazon ML' project root.
    project_root = Path(__file__).resolve().parent.parent
    dataset_dir = project_root / "dataset"

    # Define paths for all training files
    train_files = {
        "Training - Source 1": dataset_dir / "train" / "train_source1.tsv",
        "Training - Source 2": dataset_dir / "train" / "train_source2.tsv",
        "Training - Source 3": dataset_dir / "train" / "train_source3.tsv",
        "Training - Ground Truth": dataset_dir / "train" / "train_ground_truth.tsv",
    }

    # Define paths for all test files
    test_files = {
        "Test - Source 1": dataset_dir / "test" / "test_source1.tsv",
        "Test - Source 2": dataset_dir / "test" / "test_source2.tsv",
        "Test - Source 3": dataset_dir / "test" / "test_source3.tsv",
    }

    print("*" * 70)
    print(" Amazon ML Challenge: Business Entity Resolution - Data Inspection")
    print("*" * 70)

    # Inspect Training Datasets
    print("\n>>> INSPECTING TRAINING DATASETS <<<\n")
    train_status = {}
    for name, path in train_files.items():
        loaded_df = inspect_dataset(name, path)
        train_status[name] = loaded_df is not None

    # Inspect Test Datasets
    print("\n>>> INSPECTING TEST DATASETS <<<\n")
    test_status = {}
    for name, path in test_files.items():
        loaded_df = inspect_dataset(name, path)
        test_status[name] = loaded_df is not None

    # Print Summary of Found vs Missing Datasets
    print("\n" + "=" * 70)
    print(" INSPECTION SUMMARY")
    print("=" * 70)
    all_statuses = {**train_status, **test_status}
    for name, found in all_statuses.items():
        status_text = "LOADED & INSPECTED" if found else "MISSING / NOT FOUND"
        symbol = "[OK]" if found else "[!]"
        print(f"  {symbol} {name:<26}: {status_text}")
    print("=" * 70)


if __name__ == "__main__":
    main()
