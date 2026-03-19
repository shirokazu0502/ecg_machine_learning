import os
import pandas as pd
from glob import glob
import argparse

def check_data_range(data_dir):
    """
    Scans for 'dataset_*.csv' files in '0' subdirectories
    and checks if all values are within the [-0.5, 0.5] range.
    Subjects without a '0' directory or matching files are skipped.
    """
    print(f"Scanning directory: {data_dir}")
    
    # Get all subject directories
    subject_dirs = [d for d in glob(os.path.join(data_dir, '*')) if os.path.isdir(d)]
    
    if not subject_dirs:
        print("No subject directories found in the specified data_dir.")
        return

    files_to_check = []
    for subject_dir in subject_dirs:
        # Search for 'dataset_*.csv' files in the '0' subdirectory
        search_pattern = os.path.join(subject_dir, '0', 'dataset_*.csv')
        found_files = glob(search_pattern)
        files_to_check.extend(found_files)

    if not files_to_check:
        print("No 'dataset_*.csv' files found in any '0/' subdirectories.")
        return

    print(f"Found {len(files_to_check)} 'dataset_*.csv' files to check.")
    
    problematic_files = []

    for file_path in files_to_check:
        try:
            df = pd.read_csv(file_path)
            
            min_val = df.min().min()
            max_val = df.max().max()
            
            if min_val < -0.5 or max_val > 0.5:
                problematic_files.append({
                    "file": file_path,
                    "min": min_val,
                    "max": max_val
                })
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")

    print("\n--- Validation Report ---")
    if not problematic_files:
        print("Success: All values in all checked files are within the [-0.5, 0.5] range.")
    else:
        print(f"Found {len(problematic_files)} files with values outside the [-0.5, 0.5] range:")
        for info in problematic_files:
            print(f"  - File: {info['file']}")
            print(f"    Min value: {info['min']:.4f}, Max value: {info['max']:.4f}")
    print("-------------------------\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Check if values in dataset_*.csv files are within the [-0.5, 0.5] range."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/mnt/ecg_project/data/processed/for_best_resample",
        help="The base directory to search for subject folders containing dataset_*.csv."
    )
    args = parser.parse_args()
    
    check_data_range(args.data_dir)