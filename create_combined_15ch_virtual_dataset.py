import os
import sys
import pandas as pd
from glob import glob
from tqdm import tqdm

def combine_dataset_files(phys_file, virt_file, out_file):
    """
    Combines columns from a physical and a virtual electrode dataset file
    using column indices for robustness.
    """
    try:
        df_phys = pd.read_csv(phys_file)
        df_virt = pd.read_csv(virt_file)

        # --- 1. Define columns to extract ---
        
        # Virtual channels (by name, as they are consistent)
        virt_channels = [f"v_ch_{i}" for i in range(1, 10)]
        
        # Standard 12-lead ECG columns (by name)
        lead12_channels = ['A1', 'A2', 'A3', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        
        # --- 2. Extract data using column positions for physical data ---
        
        # Time is the first column (index 0)
        df_time = df_phys.iloc[:, 0:1]
        
        # Physical channels are the next 15 columns (indices 1 to 15)
        df_phys_15ch = df_phys.iloc[:, 1:16]
        
        # Get 12-lead data by name from the physical dataset file
        df_12_lead = df_phys[lead12_channels]
        
        # Get virtual channels by name from the virtual dataset file
        df_virt_9ch = df_virt[virt_channels]

        # --- 3. Combine in the specified order ---
        
        final_df = pd.concat([df_time, df_phys_15ch, df_virt_9ch, df_12_lead], axis=1)

        # --- 4. Save the combined file ---
        
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        final_df.to_csv(out_file, index=False, float_format="%.6f")

    except FileNotFoundError as e:
        print(f"Error: Input file not found - {e}", file=sys.stderr)
    except (KeyError, IndexError) as e:
        print(f"Error: Column mismatch in {os.path.basename(phys_file)} - {e}", file=sys.stderr)
    except Exception as e:
        print(f"An unexpected error occurred while processing {os.path.basename(phys_file)}: {e}", file=sys.stderr)


def process_subject_directory(phys_base_dir, virt_base_dir, out_base_dir):
    """
    Processes all dataset files in a subject's directory.
    """
    print(f"  Source (Physical): {phys_base_dir}")
    print(f"  Source (Virtual):  {virt_base_dir}")
    print(f"  Output:            {out_base_dir}")

    files_to_process = sorted(glob(os.path.join(phys_base_dir, "dataset_*.csv")))
    
    if not files_to_process:
        print(f"  No dataset files found in {os.path.basename(phys_base_dir)}.", file=sys.stderr)
        return

    for phys_file in tqdm(files_to_process, desc=f"  Combining files in {os.path.basename(phys_base_dir)}", leave=False):
        file_basename = os.path.basename(phys_file)
        virt_file = os.path.join(virt_base_dir, file_basename)
        out_file = os.path.join(out_base_dir, file_basename)
        
        combine_dataset_files(phys_file, virt_file, out_file)


def process_subject(subject_name_full, orientation):
    """
    Main execution function for a single subject with a given orientation.
    """
    print(f"\n--- Starting dataset combination for: {subject_name_full} ({orientation}) ---")

    project_root = ".."
    
    phys_data_root = os.path.join(project_root, "data", "processed", "15ch_arrange_direction", subject_name_full)
    virt_data_root = os.path.join(project_root, "data", "processed", "virtual_electrode_dataset", f"{subject_name_full}_{orientation}")
    output_root = os.path.join(project_root, "data", "processed", "combined_24ch_dataset", subject_name_full)

    # Process the main '0' directory
    print("\n  Processing main datasets...")
    process_subject_directory(
        os.path.join(phys_data_root, "0"),
        os.path.join(virt_data_root, "0"),
        os.path.join(output_root, "0")
    )

    # Process the 'moving_ave_datasets' subdirectory
    print("\n  Processing moving average datasets...")
    process_subject_directory(
        os.path.join(phys_data_root, "0", "moving_ave_datasets"),
        os.path.join(virt_data_root, "0", "moving_ave_datasets"),
        os.path.join(output_root, "0", "moving_ave_datasets")
    )

    print(f"\n--- Finished processing for: {subject_name_full} ---")


if __name__ == "__main__":
    # Process 'flipped' subjects first
    flipped_subjects = [
        "asano_0714_0.8s", "ikejima_0714_0.8s", "kanda_0807_0.8s",
        "nakashimizu_0512_0.8s", "nishio_0513_0.8s", "noda_0714_0.8s",
        "takahashi_jr_0512_0.8s", "gosha_0807_0.8s",
    ]
    print("===== PROCESSING 'FLIPPED' SUBJECTS =====")
    for subject in flipped_subjects:
        process_subject(subject, "flipped")

    # Process 'normal' subjects next
    normal_subjects = [
        "goto_1219_0.8s", "kawai_1115_0.8s", "matumoto_1128_0.8s",
        "patient4_1001_0.8s", "patient6_1001_0.8s", "patient8_1109_0.8s",
        "patient9_1109_0.8s", "taniguchi_1107_0.8s", "yoshikura_1130_0.8s",
    ]
    print("\n===== PROCESSING 'NORMAL' SUBJECTS =====")
    for subject in normal_subjects:
        process_subject(subject, "normal")
    
    print("\nAll processing complete.")