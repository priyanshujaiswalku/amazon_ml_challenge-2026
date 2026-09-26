#!/usr/bin/env python3
"""
Submission Packaging Utility
=============================
Phase 6.7: Automated, Validated Packaging for Amazon ML Challenge 2026.

Packages final candidate sets, matching predictions, runnable pipeline source code,
reproduction instructions, pinned dependencies, and filled methodology documentation
into the official competition zip archive format:

    <team_name>_submission.zip
    ├── output/
    │   ├── matching_results.tsv        # Evaluated on competition leaderboard
    │   └── candidate_pairs.tsv         # Blocking candidate set
    ├── code/
    │   └── business_entity_resolution/
    │       ├── src/                    # Pipeline source code modules
    │       ├── README.md               # End-to-end reproduction guide
    │       └── requirements.txt        # Pinned runtime dependencies
    └── Documentation_template.md       # Contest methodology write-up
"""

import argparse
import os
from pathlib import Path
import subprocess
import sys
import zipfile
from typing import Dict, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def run_pre_packaging_validation(
    test_dir: Path,
    matching_file: Path,
    candidate_file: Path,
) -> bool:
    """
    Executes the official submission validator prior to archive generation.

    Args:
        test_dir: Directory containing test source TSVs.
        matching_file: Path to output/matching_results.tsv.
        candidate_file: Path to output/candidate_pairs.tsv.

    Returns:
        bool: True if validation passed (exit code 0), False otherwise.
    """
    validator_script = PROJECT_ROOT / "utils" / "validate_submission.py"
    if not validator_script.exists():
        print(f"[ERROR] Validator script not found: {validator_script}", file=sys.stderr)
        return False

    cmd = [
        sys.executable,
        str(validator_script),
        "--matching", str(matching_file),
        "--candidate", str(candidate_file),
        "--test-dir", str(test_dir),
        "--check-ids",
    ]

    print("\n" + "=" * 75)
    print("  Pre-Packaging Validation Audit (utils/validate_submission.py)")
    print("=" * 75)
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)

    if result.returncode != 0:
        print("[FAIL] Submission files failed official validation check!", file=sys.stderr)
        return False

    print("[PASS] Pre-packaging validation passed successfully.")
    return True


def package_submission(
    team_name: str = "team_priyanshu",
    output_zip: Optional[Path] = None,
    skip_validation: bool = False,
    test_dir: Optional[Path] = None,
) -> Path:
    """
    Constructs the compliant submission zip archive.

    Args:
        team_name: Team identifier used in archive filename prefix.
        output_zip: Destination path for the zip archive.
        skip_validation: Whether to bypass validate_submission.py check.
        test_dir: Test dataset directory.

    Returns:
        Path: Path to the generated zip archive.

    Raises:
        FileNotFoundError: If any mandatory submission component is missing.
        RuntimeError: If pre-packaging validation fails.
    """
    test_dir = test_dir or (PROJECT_ROOT / "dataset" / "test")
    matching_path = PROJECT_ROOT / "output" / "matching_results.tsv"
    candidate_path = PROJECT_ROOT / "output" / "candidate_pairs.tsv"
    doc_path = PROJECT_ROOT / "Documentation_template.md"
    readme_path = PROJECT_ROOT / "README.md"
    reqs_path = PROJECT_ROOT / "requirements.txt"
    src_dir = PROJECT_ROOT / "src"

    # Verify existence of required components
    required_files = [
        (matching_path, "Final matching results TSV"),
        (candidate_path, "Candidate blocking pairs TSV"),
        (doc_path, "Methodology document (Documentation_template.md)"),
        (readme_path, "Pipeline reproduction guide (README.md)"),
        (reqs_path, "Pinned requirements (requirements.txt)"),
    ]
    for path, description in required_files:
        if not path.exists():
            raise FileNotFoundError(f"Missing required submission component: {description} ({path})")

    if not src_dir.exists() or not any(src_dir.glob("*.py")):
        raise FileNotFoundError(f"Source directory missing or empty: {src_dir}")

    # Run official validator unless explicitly skipped
    if not skip_validation:
        valid = run_pre_packaging_validation(test_dir, matching_path, candidate_path)
        if not valid:
            raise RuntimeError("Validation failed. Halting packaging to avoid invalid submission.")

    # Determine destination zip file path
    clean_team_name = team_name.strip().replace(" ", "_").lower()
    if output_zip is None:
        output_zip = PROJECT_ROOT / f"{clean_team_name}_submission.zip"
    else:
        output_zip = Path(output_zip)

    output_zip.parent.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 75)
    print(f"  Constructing Submission Archive: {output_zip.name}")
    print("=" * 75)

    # Archive file mapping specification
    # Target archive path -> Source file on disk
    archive_manifest: List[Tuple[str, Path]] = [
        ("output/matching_results.tsv", matching_path),
        ("output/candidate_pairs.tsv", candidate_path),
        ("Documentation_template.md", doc_path),
        ("code/business_entity_resolution/README.md", readme_path),
        ("code/business_entity_resolution/requirements.txt", reqs_path),
    ]

    # Include all Python source files in src/
    for src_file in sorted(src_dir.glob("*.py")):
        archive_manifest.append(
            (f"code/business_entity_resolution/src/{src_file.name}", src_file)
        )

    # Write Zip Archive with compression
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for archive_name, src_file in archive_manifest:
            print(f"  Adding: {archive_name:<50} ({src_file.stat().st_size:,} bytes)")
            zf.write(src_file, arcname=archive_name)

    # Verify Archive Contents
    print("\n" + "-" * 75)
    print(f"  Archive Verification Table ({output_zip.name})")
    print("-" * 75)
    with zipfile.ZipFile(output_zip, "r") as zf:
        infolist = zf.infolist()
        total_uncompressed = 0
        for info in infolist:
            total_uncompressed += info.file_size
            ratio = (1.0 - (info.compress_size / info.file_size)) * 100 if info.file_size > 0 else 0
            print(f"  {info.filename:<50} {info.file_size:>8,} bytes  [{ratio:>5.1f}% compressed]")

    zip_size_bytes = output_zip.stat().st_size
    print("-" * 75)
    print(f"  Total Files in Archive:  {len(infolist)}")
    print(f"  Uncompressed Size:       {total_uncompressed:,} bytes")
    print(f"  Compressed Archive Size: {zip_size_bytes:,} bytes ({zip_size_bytes / 1024:.1f} KB)")
    print(f"  Archive Location:        {output_zip.resolve()}")
    print("=" * 75)
    print(f"[SUCCESS] Submission package successfully built and verified: {output_zip.name}\n")

    return output_zip


def main():
    parser = argparse.ArgumentParser(description="Step 6.7: Package Amazon ML Challenge Submission Zip")
    parser.add_argument(
        "--team-name",
        type=str,
        default="team_priyanshu",
        help="Team name for zip prefix (e.g. team_priyanshu)",
    )
    parser.add_argument(
        "--output-zip",
        type=str,
        default=None,
        help="Optional destination path for the submission zip",
    )
    parser.add_argument(
        "--test-dir",
        type=str,
        default=None,
        help="Test dataset directory (default: dataset/test)",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip pre-packaging validate_submission.py check",
    )
    args = parser.parse_args()

    test_dir = Path(args.test_dir) if args.test_dir else None
    output_zip = Path(args.output_zip) if args.output_zip else None

    try:
        package_submission(
            team_name=args.team_name,
            output_zip=output_zip,
            skip_validation=args.skip_validation,
            test_dir=test_dir,
        )
    except Exception as e:
        print(f"\n[ERROR] Packaging failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
