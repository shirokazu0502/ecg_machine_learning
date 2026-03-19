import os
import sys
import argparse
from glob import glob
import pandas as pd
import numpy as np
import re
from collections import defaultdict

# Add the project root to the Python path to import utils
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

from settings import OUTPUT_DIR
import utils


def find_target_name_from_dir(dir_name):
    """Extracts TARGETNAME from a directory name using regex."""
    match = re.search(r"TARGETNAME=([a-zA-Z0-9_]+)", dir_name)
    if match:
        return match.group(1)
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Calculate Weighted Diagnostic Distortion (WDD) by concatenating all waveforms for each subject."
    )
    parser.add_argument(
        "--results_root_dir",
        type=str,
        required=True,
        help="Path to the root directory containing multiple experiment folders (e.g., 'outputs/fig_newref/cnn_grid_search_...').",
    )
    args = parser.parse_args()

    # Construct the full path to the results directory
    dataset_path = os.path.join(OUTPUT_DIR, "figs_newref", args.results_root_dir)

    if not os.path.isdir(dataset_path):
        print(f"Error: Root directory not found at '{dataset_path}'")
        sys.exit(1)

    print(f"Scanning for test directories in: {dataset_path}")

    # Recursively find all directories that correspond to a 'test' run
    test_dirs = [
        d
        for d in glob(os.path.join(dataset_path, "**/"), recursive=True)
        if "_test_TARGETNAME=" in os.path.basename(os.path.normpath(d))
        and os.path.isdir(d)
    ]

    if not test_dirs:
        print("No test directories found matching the pattern `_test_TARGETNAME=`.")
        sys.exit(0)

    print(f"Found {len(test_dirs)} total test directories to process.")

    # Group directories and their waveform files by TARGETNAME
    grouped_files = defaultdict(lambda: {"orig": [], "recon": []})

    for exp_dir in test_dirs:
        target_name = find_target_name_from_dir(
            os.path.basename(os.path.normpath(exp_dir))
        )
        if not target_name:
            continue

        waveforms_path = os.path.join(exp_dir, "waveforms")
        if not os.path.isdir(waveforms_path):
            continue

        # Find all waveform files for this target's test run
        for recon_file in glob(os.path.join(waveforms_path, "*_reconx.csv")):
            base_name = os.path.basename(recon_file).replace("_reconx.csv", "")
            orig_file = os.path.join(waveforms_path, f"{base_name}_xo.csv")

            if os.path.exists(orig_file):
                grouped_files[target_name]["orig"].append(orig_file)
                grouped_files[target_name]["recon"].append(recon_file)

    print(f"Grouped files for {len(grouped_files)} unique TARGETNAMEs.")

    all_results = []
    sampling_rate = 500  # Assuming fixed sampling rate

    for target_name, files in grouped_files.items():
        print(f"\nProcessing TARGETNAME: {target_name}")

        if not files["orig"] or not files["recon"]:
            print("  - Missing original or reconstructed files. Skipping.")
            continue

        try:
            # Load and concatenate all waveforms for the current target
            orig_waveforms = [
                pd.read_csv(f)["A2"].to_numpy() for f in sorted(files["orig"])
            ]
            recon_waveforms = [
                pd.read_csv(f)["A2"].to_numpy() for f in sorted(files["recon"])
            ]

            long_orig_waveform = np.concatenate(orig_waveforms)
            long_recon_waveform = np.concatenate(recon_waveforms)

            print(
                f"  - Concatenated {len(files['orig'])} files into a single waveform of length {len(long_orig_waveform)}."
            )

            # Extract features from the long, continuous signals
            features_orig = utils.extract_wdd_features(
                long_orig_waveform, sampling_rate=sampling_rate
            )
            features_recon = utils.extract_wdd_features(
                long_recon_waveform, sampling_rate=sampling_rate
            )

            # Calculate a single WDD score for the concatenated waveform
            wdd_score = utils.calculate_wdd(features_orig, features_recon)

            if not np.isnan(wdd_score):
                all_results.append({"target_name": target_name, "wdd_score": wdd_score})
                print(f"  - Calculated WDD for {target_name}: {wdd_score:.6f}")
            else:
                print(f"  - WDD calculation resulted in NaN for {target_name}.")

        except Exception as e:
            print(f"An error occurred while processing {target_name}: {e}")

    if not all_results:
        print("\nCould not calculate WDD for any subjects.")
        return

    # --- Summarize and Save Final Results ---
    df_summary = pd.DataFrame(all_results)

    mean_wdd_total = df_summary["wdd_score"].mean()
    std_wdd_total = df_summary["wdd_score"].std()

    print("\n\n--- Overall WDD Evaluation Summary ---")
    print(f"Average WDD Score across all subjects: {mean_wdd_total:.6f}")
    print(f"Standard Deviation of WDD Scores: {std_wdd_total:.6f}")

    # Save the summary to a new CSV file in the root directory
    output_csv_path = os.path.join(dataset_path, "wdd_evaluation_summary.csv")
    df_summary.sort_values(by="target_name", inplace=True)
    df_summary.to_csv(output_csv_path, index=False)

    print(f"\nSummary of WDD scores saved to: {output_csv_path}")
    print(df_summary)


if __name__ == "__main__":
    main()
