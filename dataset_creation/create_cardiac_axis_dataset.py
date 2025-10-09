
import os
import sys
import argparse
import pandas as pd
import numpy as np
import glob
from tqdm import tqdm

# Add base directory to sys.path
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(base_dir)

# --- Constants ---
# The QRS complex is centered around index 150 in the source data.
# We define a window around it to calculate the integral.
QRS_WINDOW_START = 100
QRS_WINDOW_END = 200

def calculate_cardiac_axis(df, col_d, col_m, col_p):
    """
    Calculates the cardiac axis angle from the signals of three orthogonal points.

    Args:
        df (pd.DataFrame): The dataframe for a single heartbeat.
        col_d (str): The column name for the Lower Left position.
        col_m (str): The column name for the Upper Right position.
        col_p (str): The column name for the Lower Right position.

    Returns:
        float: The calculated cardiac axis angle in degrees.
    """
    # Ensure columns exist
    for col in [col_d, col_m, col_p]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in the dataframe.")

    # Define orthogonal vectors based on the square geometry
    # Vertical Component (Y-axis): M (Upper Right) -> P (Lower Right)
    signal_v = df[col_p] - df[col_m]
    
    # Horizontal Component (X-axis): P (Lower Right) -> D (Lower Left)
    signal_h = df[col_d] - df[col_p]

    # Integrate over the QRS window
    qrs_integral_v = np.sum(signal_v[QRS_WINDOW_START:QRS_WINDOW_END])
    qrs_integral_h = np.sum(signal_h[QRS_WINDOW_START:QRS_WINDOW_END])

    # Calculate the angle using arctan2(y, x)
    # The result is in radians. We convert it to degrees.
    angle_rad = np.arctan2(qrs_integral_v, qrs_integral_h)
    angle_deg = np.degrees(angle_rad)

    return angle_deg

def process_subject(subject_dir, col_d, col_m, col_p):
    """
    Processes all heartbeat CSVs for a single subject to add the cardiac axis.
    """
    source_data_path = os.path.join(subject_dir, "0", "moving_ave_datasets")
    if not os.path.isdir(source_data_path):
        print(f"Error: 'moving_ave_datasets' directory not found in {subject_dir}")
        return

    # Define output path
    subject_name = os.path.basename(subject_dir)
    output_dir = os.path.join(base_dir, "data", "processed", "cardiac_axis_dataset", subject_name, "0", "moving_ave_datasets")
    os.makedirs(output_dir, exist_ok=True)

    # Find all dataset files
    files_to_process = sorted(glob.glob(os.path.join(source_data_path, "dataset_*.csv")))
    if not files_to_process:
        print(f"No dataset CSVs found for subject {subject_name}")
        return

    print(f"Processing {len(files_to_process)} files for subject: {subject_name}")
    
    for file_path in tqdm(files_to_process, desc=f"Processing {subject_name}"):
        try:
            df = pd.read_csv(file_path)
            
            # Calculate cardiac axis
            cardiac_axis = calculate_cardiac_axis(df, col_d, col_m, col_p)
            
            # Add the new column
            df['cardiac_axis'] = cardiac_axis
            
            # Save to new location
            output_filename = os.path.basename(file_path)
            output_filepath = os.path.join(output_dir, output_filename)
            df.to_csv(output_filepath, index=False)

        except Exception as e:
            print(f"Could not process file {file_path}. Error: {e}")
            continue
            
    print(f"Finished processing for {subject_name}. New dataset saved in {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Create a new dataset with cardiac axis information.")
    parser.add_argument("--subject_dir", type=str, required=True, help="Path to the subject's data directory (e.g., 'data/processed/15ch_arrange_direction/asano_0714_0.8s').")
    parser.add_argument("--col_d", type=str, required=True, help="Column name for the Lower Left position (e.g., 'ch_4').")
    parser.add_argument("--col_m", type=str, required=True, help="Column name for the Upper Right position (e.g., 'ch_13').")
    parser.add_argument("--col_p", type=str, required=True, help="Column name for the Lower Right position (e.g., 'ch_16').")
    
    args = parser.parse_args()
    
    # Correct base_dir to be project root
    project_root = os.path.dirname(base_dir)
    subject_full_path = os.path.join(project_root, args.subject_dir)

    process_subject(subject_full_path, args.col_d, args.col_m, args.col_p)

if __name__ == "__main__":
    main()
