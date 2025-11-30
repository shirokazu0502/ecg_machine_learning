import os
import sys
import pandas as pd
import random
from glob import glob
from tqdm import tqdm
import subprocess

def create_differential_file(input_file, output_file, base_ch):
    """
    Creates a 15-channel differential dataset by subtracting a base channel.
    """
    try:
        df = pd.read_csv(input_file)

        # --- 1. Define columns ---
        all_phys_channels = [f"ch_{i}" for i in range(1, 17)]
        other_channels = [ch for ch in all_phys_channels if ch != base_ch]
        lead12_channels = ['A1', 'A2', 'A3', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        time_col = ['Time']

        # --- 2. Perform subtraction ---
        if base_ch not in df.columns:
            print(f"Error: Base channel '{base_ch}' not found in {input_file}", file=sys.stderr)
            return
            
        base_series = df[base_ch]
        diff_df = df[other_channels].subtract(base_series, axis=0)

        # --- 3. Combine and save ---
        df_time = df[time_col]
        df_12_lead = df[lead12_channels]
        
        final_df = pd.concat([df_time, diff_df, df_12_lead], axis=1)
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        final_df.to_csv(output_file, index=False, float_format="%.6f")

    except FileNotFoundError:
        print(f"Error: Input file not found - {input_file}", file=sys.stderr)
    except (KeyError, IndexError) as e:
        print(f"Error: Column mismatch in {os.path.basename(input_file)} - {e}", file=sys.stderr)
    except Exception as e:
        print(f"An unexpected error occurred while processing {os.path.basename(input_file)}: {e}", file=sys.stderr)


def process_subject(subject_name, base_ch, input_root, output_root):
    """
    Processes all data for a single subject.
    """
    print(f"\n--- Processing: {subject_name} (Base: {base_ch}) ---")
    
    # Define paths for this subject
    input_dir = os.path.join(input_root, subject_name)
    output_dir = os.path.join(output_root, subject_name)

    # Process main '0' directory
    print("  Processing main datasets...")
    main_input_path = os.path.join(input_dir, "0")
    main_output_path = os.path.join(output_dir, "0")
    files_to_process = sorted(glob(os.path.join(main_input_path, "dataset_*.csv")))
    if files_to_process:
        for input_file in tqdm(files_to_process, desc="  Main files", leave=False):
            output_file = os.path.join(main_output_path, os.path.basename(input_file))
            create_differential_file(input_file, output_file, base_ch)
    else:
        print(f"  No main dataset files found for {subject_name}.")

    # Process 'moving_ave_datasets' subdirectory
    print("  Processing moving average datasets...")
    ma_input_path = os.path.join(main_input_path, "moving_ave_datasets")
    ma_output_path = os.path.join(main_output_path, "moving_ave_datasets")
    ma_files_to_process = sorted(glob(os.path.join(ma_input_path, "dataset_*.csv")))
    if ma_files_to_process:
        for input_file in tqdm(ma_files_to_process, desc="  Moving avg", leave=False):
            output_file = os.path.join(ma_output_path, os.path.basename(input_file))
            create_differential_file(input_file, output_file, base_ch)
    else:
        print(f"  No moving average dataset files found for {subject_name}.")


if __name__ == "__main__":
    project_root = ".."
    input_root = os.path.join(project_root, "data", "processed", "for_best_resample")
    output_root = os.path.join(project_root, "data", "processed", "no_orientation_correction_dataset")
    
    # Get list of subjects from the input directory
    all_items = os.listdir(input_root)
    subject_list = sorted([item for item in all_items if os.path.isdir(os.path.join(input_root, item))])

    if not subject_list:
        print("No subject directories found. Exiting.")
        sys.exit(0)

    print(f"Found {len(subject_list)} subjects to process.")

    base_ch_log = []
    base_ch_options = [1, 4, 13, 16]

    # Process each subject
    for subject_name in subject_list:
        # Randomly choose a base channel number for this subject
        random_base_ch_num = random.choice(base_ch_options)
        random_base_ch_name = f"ch_{random_base_ch_num}"

        base_ch_log.append({
            "subject_name": subject_name,
            "random_base_channel": random_base_ch_name
        })
        process_subject(subject_name, random_base_ch_name, input_root, output_root)

    # Save the log file
    log_df = pd.DataFrame(base_ch_log)
    log_file_path = os.path.join(output_root, "random_base_channel_log.csv")
    os.makedirs(output_root, exist_ok=True)
    log_df.to_csv(log_file_path, index=False)

    print(f"\nProcessing complete. Base channel log saved to: {log_file_path}")
