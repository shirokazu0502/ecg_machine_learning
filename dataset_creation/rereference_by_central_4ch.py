# -*- coding: utf-8 -*-
"""
This script performs re-referencing on 16-channel ECG data using a virtual
electrode calculated from the four central channels (ch_6, 7, 10, 11).

It has special handling for patient directories:
- If 'center_*' subdirectories are found, it aggregates them into a single
  '0' directory, ignoring all other subdirectories.
- If no 'center_*' subdirectories are found, it processes each subdirectory
  individually, ignoring any containing '15ch'.

Conditional channel reversal is applied to data from 'center_1' and 'center_2'.
"""

import os
import glob
import pandas as pd
import numpy as np

# --- Configuration ---
BASE_INPUT_DIR = "/mnt/ecg_project/data/processed/for_best_resample/"
BASE_OUTPUT_DIR = "/mnt/ecg_project/data/processed/reference_center_4ch/"

# --- Channel Definitions ---
ALL_16_CHANNELS = [f"ch_{i+1}" for i in range(16)]
CENTRAL_4_CHANNELS = ["ch_6", "ch_7", "ch_10", "ch_11"]
REVERSED_16_CHANNELS = [f"ch_{i}" for i in range(16, 0, -1)]

# --- Reversal Conditions ---
REVERSAL_PATIENT_NAMES = [
    "goto",
    "matumoto",
    "yoshikura",
    "taniguchi",
    "kawai",
    "patient1",
    "patient2",
    "patient3",
    "patient4",
    "patient5",
    "patient6",
    "patient7",
    "patient8",
    "patient9",
    "patient10",
]


def process_single_file(file_path, output_path, perform_reversal):
    """
    Reads a single CSV, re-references it, applies reversal if needed, and saves it.
    Returns True on success, False on failure.
    """
    try:
        df = pd.read_csv(file_path)
        # Clean column names by stripping whitespace and check again
        df.columns = df.columns.str.strip()
        if not set(ALL_16_CHANNELS).issubset(df.columns):
            print(
                f"        - Skipping {os.path.basename(file_path)}: Missing required 16ch columns."
            )
            return False

        virtual_electrode = df[CENTRAL_4_CHANNELS].mean(axis=1)
        df_rereferenced = df[ALL_16_CHANNELS].subtract(virtual_electrode, axis=0)

        output_df = df.copy()
        output_df[ALL_16_CHANNELS] = df_rereferenced

        if perform_reversal:
            time_col = ["Time"] if "Time" in df.columns else []
            medical_cols = [
                col
                for col in df.columns
                if col not in ALL_16_CHANNELS and col != "Time"
            ]
            new_order = time_col + REVERSED_16_CHANNELS + medical_cols
            if set(new_order) == set(df.columns):
                output_df = output_df[new_order]
            else:
                print(
                    f"        - WARNING: Column set mismatch during reversal for {file_path}. Saving in original order."
                )

        output_df.to_csv(output_path, index=False)
        return True
    except Exception as e:
        print(
            f"        - ERROR: Failed to process {os.path.basename(file_path)}. Error: {e}"
        )
        return False


def recreate_ponset_file(source_path, output_path):
    """Reads a ponset file and writes it to the destination."""
    try:
        content_df = pd.read_csv(source_path)
        content_df.to_csv(output_path, index=False)
        return True
    except Exception as e:
        print(
            f"      - ERROR: Could not recreate {os.path.basename(source_path)}. Error: {e}"
        )
        return False


def process_directory_content(input_dir, output_dir, perform_reversal):
    """
    Processes all files in a single directory without renumbering.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Process dataset files
    dataset_files = sorted(glob.glob(os.path.join(input_dir, "dataset_*.csv")))
    print(
        f"    - Found {len(dataset_files)} dataset files in {os.path.basename(input_dir)}."
    )
    for file_path in dataset_files:
        output_file = os.path.join(output_dir, os.path.basename(file_path))
        process_single_file(file_path, output_file, perform_reversal)

    # Recreate ponset files
    ponset_files = sorted(glob.glob(os.path.join(input_dir, "ponset_toffset_*.csv")))
    print(
        f"    - Found {len(ponset_files)} ponset_toffset files in {os.path.basename(input_dir)}."
    )
    for file_path in ponset_files:
        output_file = os.path.join(output_dir, os.path.basename(file_path))
        recreate_ponset_file(file_path, output_file)


def aggregate_center_dirs(center_dirs, patient_name):
    """
    Aggregates all center_* directories into a single '0' directory (instead of center_all),
    with sequential renaming and conditional reversal.
    Also, records the patient name in a manifest file.
    """
    print(f"  + Aggregating {len(center_dirs)} 'center_*' directories into '0'")

    # Define output paths for the aggregated '0' directory
    aggregated_output_path = os.path.join(BASE_OUTPUT_DIR, patient_name, "0")
    aggregated_moving_ave_output_path = os.path.join(
        aggregated_output_path, "moving_ave_datasets"
    )
    os.makedirs(aggregated_output_path, exist_ok=True)  # Ensure '0' directory exists
    os.makedirs(
        aggregated_moving_ave_output_path, exist_ok=True
    )  # Ensure '0/moving_ave_datasets' exists

    dataset_counter = 0
    moving_ave_counter = 0

    for source_dir_path in center_dirs:
        source_dir_name = os.path.basename(source_dir_path)
        print(f"    - Reading from {source_dir_name}...")

        perform_reversal = (
            "center_1" in source_dir_name or "center_2" in source_dir_name
        ) or any(name in patient_name for name in REVERSAL_PATIENT_NAMES)

        # Process main dataset files
        dataset_files = sorted(
            glob.glob(os.path.join(source_dir_path, "dataset_*.csv"))
        )
        for file_path in dataset_files:
            output_file_path = os.path.join(
                aggregated_output_path, f"dataset_{dataset_counter:03d}.csv"
            )
            if process_single_file(file_path, output_file_path, perform_reversal):
                ponset_name = (
                    "ponset_toffset_" + os.path.basename(file_path).split("_")[1]
                )
                source_ponset_path = os.path.join(source_dir_path, ponset_name)
                if os.path.exists(source_ponset_path):
                    dest_ponset_path = os.path.join(
                        aggregated_output_path,
                        f"ponset_toffset_{dataset_counter:03d}.csv",
                    )
                    recreate_ponset_file(source_ponset_path, dest_ponset_path)
                dataset_counter += 1

        # Process moving_ave_datasets
        moving_ave_source_path = os.path.join(source_dir_path, "moving_ave_datasets")
        if os.path.isdir(moving_ave_source_path):
            moving_ave_files = sorted(
                glob.glob(os.path.join(moving_ave_source_path, "dataset_*.csv"))
            )
            for file_path in moving_ave_files:
                output_file_path = os.path.join(
                    aggregated_moving_ave_output_path,
                    f"dataset_{moving_ave_counter:03d}.csv",
                )
                if process_single_file(file_path, output_file_path, perform_reversal):
                    ponset_name = (
                        "ponset_toffset_" + os.path.basename(file_path).split("_")[1]
                    )
                    source_ponset_path = os.path.join(
                        moving_ave_source_path, ponset_name
                    )
                    if os.path.exists(source_ponset_path):
                        dest_ponset_path = os.path.join(
                            aggregated_moving_ave_output_path,
                            f"ponset_toffset_{moving_ave_counter:03d}.csv",
                        )
                        recreate_ponset_file(source_ponset_path, dest_ponset_path)
                    moving_ave_counter += 1

    # Record this patient in the manifest file
    manifest_file_path = os.path.join(BASE_OUTPUT_DIR, "aggregated_patients.txt")
    with open(manifest_file_path, "a") as f:
        f.write(patient_name + "\n")
    print(f"  > Recorded {patient_name} in manifest file.")


def process_patient_directory(patient_dir):
    """
    Decides whether to aggregate 'center_*' dirs or process others individually.
    """
    patient_name = os.path.basename(patient_dir)
    print(f"--- Processing Patient: {patient_name} ---")

    all_subdirs = sorted(
        [d for d in glob.glob(os.path.join(patient_dir, "*")) if os.path.isdir(d)]
    )

    if not all_subdirs:
        print(f"  > No subdirectories found for patient {patient_name}. Skipping.")
        return

    # Categorize directories
    center_dirs = [d for d in all_subdirs if os.path.basename(d).startswith("center_")]
    other_dirs = [
        d
        for d in all_subdirs
        if not os.path.basename(d).startswith("center_")
        and "15ch" not in os.path.basename(d)
    ]

    if center_dirs:
        # If any center_* dirs exist, only process them and aggregate.
        aggregate_center_dirs(center_dirs, patient_name)
    else:
        # If no center_* dirs, process each non-15ch directory individually.
        for source_dir_path in other_dirs:
            source_dir_name = os.path.basename(source_dir_path)
            print(f"  + Processing individual subdirectory: {source_dir_name}")

            output_dir_path = os.path.join(
                BASE_OUTPUT_DIR, patient_name, source_dir_name
            )
            perform_reversal = any(
                name in patient_name for name in REVERSAL_PATIENT_NAMES
            )

            # Process main directory content
            process_directory_content(
                source_dir_path, output_dir_path, perform_reversal
            )

            # Process moving_ave_datasets subdirectory if it exists
            moving_ave_source = os.path.join(source_dir_path, "moving_ave_datasets")
            if os.path.isdir(moving_ave_source):
                print(
                    "    -> Found and processing 'moving_ave_datasets' subdirectory..."
                )
                moving_ave_output = os.path.join(output_dir_path, "moving_ave_datasets")
                process_directory_content(
                    moving_ave_source, moving_ave_output, perform_reversal
                )

    print(f"  > Finished processing for {patient_name}.\n")


if __name__ == "__main__":
    print("Starting Re-referencing Process...")
    patient_dirs = sorted(
        [d for d in glob.glob(os.path.join(BASE_INPUT_DIR, "*")) if os.path.isdir(d)]
    )

    if not patient_dirs:
        print(f"No patient directories found in {BASE_INPUT_DIR}. Exiting.")
    else:
        print(f"Found {len(patient_dirs)} patient directories to analyze.")
        for patient_dir in patient_dirs:
            process_patient_directory(patient_dir)

    print("--- Re-referencing Process Finished ---")
