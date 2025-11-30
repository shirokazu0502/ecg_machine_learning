import os
import sys
import pandas as pd
import numpy as np
from glob import glob
from tqdm import tqdm

def create_differential_file(input_file, output_file, base_ch):
    """
    Creates a 15-channel differential dataset by subtracting a base channel.
    """
    try:
        df = pd.read_csv(input_file)

        all_phys_channels = [f"ch_{i}" for i in range(1, 17)]
        if base_ch not in all_phys_channels:
            print(f"Error: Invalid base_ch '{base_ch}' specified.", file=sys.stderr)
            return
            
        other_channels = [ch for ch in all_phys_channels if ch != base_ch]
        lead12_channels = ['A1', 'A2', 'A3', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        time_col = ['Time']

        if base_ch not in df.columns:
            print(f"Error: Base channel '{base_ch}' not found in {input_file}", file=sys.stderr)
            return
            
        base_series = df[base_ch]
        diff_df = df[other_channels].subtract(base_series, axis=0)
        
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


def get_base_channel_from_orientation(orientation):
    """
    Determines which channel corresponds to the 'anatomical top-left'
    based on the estimated orientation of the grid.
    This mapping assumes the 'flipped' layout (ch_1 top-left) is the target 'upright' orientation.
    """
    if orientation == "Down-Right (Likely Flipped)":
        # Grid is likely upright, ch_1 is top-left.
        return "ch_1"
    elif orientation == "Up-Left (Likely Normal)":
        # Grid is likely upside-down, ch_16 is top-left.
        return "ch_16"
    elif orientation == "Up-Right":
        # Grid is likely rotated -90 degrees, ch_4 is top-left.
        return "ch_4"
    elif orientation == "Down-Left":
        # Grid is likely rotated +90 degrees, ch_13 is top-left.
        return "ch_13"
    else:
        # For uncertain or ambiguous cases, default to ch_1 and log it.
        return "ch_1"


def main():
    project_root = ".."
    input_root = os.path.join(project_root, "data", "processed", "for_best_resample_base_max_amp")
    output_root = os.path.join(project_root, "data", "processed", "auto_orientation_corrected_dataset_max_amp")
    orientation_file = os.path.join(project_root, "data", "processed", "estimated_orientations_max_amp.csv")

    # 1. Load orientation data
    try:
        orientation_df = pd.read_csv(orientation_file)
        # Create a dictionary for easy lookup, dropping subjects with errors
        orientation_map = orientation_df.dropna(subset=['estimated_orientation']).set_index('subject')['estimated_orientation'].to_dict()
    except FileNotFoundError:
        print(f"Error: Orientation file not found at {orientation_file}", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded orientation data for {len(orientation_map)} subjects.")

    # 2. Process each subject based on the map
    for subject_name, orientation in tqdm(orientation_map.items(), desc="Applying Correction"):
        
        base_ch = get_base_channel_from_orientation(orientation)
        print(f"\n--- Processing: {subject_name} ---")
        print(f"  - Estimated Orientation: {orientation}")
        print(f"  - Selected Base Channel: {base_ch}")

        # Define paths
        subject_input_dir = os.path.join(input_root, subject_name)
        subject_output_dir = os.path.join(output_root, subject_name)

        # Process main '0' directory
        main_input_path = os.path.join(subject_input_dir, "0")
        main_output_path = os.path.join(subject_output_dir, "0")
        files_to_process = sorted(glob(os.path.join(main_input_path, "dataset_*.csv")))
        
        if files_to_process:
            for input_file in files_to_process:
                output_file = os.path.join(main_output_path, os.path.basename(input_file))
                create_differential_file(input_file, output_file, base_ch)
        else:
            print(f"  - No main dataset files found for {subject_name}.")

        # Process 'moving_ave_datasets' subdirectory
        ma_input_path = os.path.join(main_input_path, "moving_ave_datasets")
        ma_output_path = os.path.join(main_output_path, "moving_ave_datasets")
        ma_files_to_process = sorted(glob(os.path.join(ma_input_path, "dataset_*.csv")))

        if ma_files_to_process:
            for input_file in ma_files_to_process:
                output_file = os.path.join(ma_output_path, os.path.basename(input_file))
                create_differential_file(input_file, output_file, base_ch)
        else:
            print(f"  - No moving average dataset files found for {subject_name}.")

    print("\nProcessing complete.")
    print(f"Corrected datasets saved in: {output_root}")


if __name__ == "__main__":
    main()
