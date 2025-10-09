
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
# The R-peak is known to be at index 150.
R_PEAK_INDEX = 150

def calculate_axis_integral(signal_h, signal_v):
    """Calculates cardiac axis based on signal area (integral)."""
    qrs_integral_v = np.sum(signal_v[QRS_WINDOW_START:QRS_WINDOW_END])
    qrs_integral_h = np.sum(signal_h[QRS_WINDOW_START:QRS_WINDOW_END])
    angle_rad = np.arctan2(qrs_integral_v, qrs_integral_h)
    return np.degrees(angle_rad)

def get_net_amplitude(signal_segment):
    """Finds Q, R, S amplitudes and calculates net amplitude."""
    # R is the max value in the segment
    r_amp = np.max(signal_segment)
    r_index = np.argmax(signal_segment)

    # Q is the min value before the R peak
    q_segment = signal_segment[:r_index]
    q_amp = np.min(q_segment) if len(q_segment) > 0 else 0

    # S is the min value after the R peak
    s_segment = signal_segment[r_index+1:]
    s_amp = np.min(s_segment) if len(s_segment) > 0 else 0
    
    # Net amplitude calculation
    net_amplitude = r_amp - abs(q_amp) - abs(s_amp)
    return net_amplitude

def calculate_axis_amplitude(signal_h, signal_v):
    """Calculates cardiac axis based on QRS amplitude."""
    qrs_segment_h = signal_h[QRS_WINDOW_START:QRS_WINDOW_END].to_numpy()
    qrs_segment_v = signal_v[QRS_WINDOW_START:QRS_WINDOW_END].to_numpy()

    net_amp_h = get_net_amplitude(qrs_segment_h)
    net_amp_v = get_net_amplitude(qrs_segment_v)

    angle_rad = np.arctan2(net_amp_v, net_amp_h)
    return np.degrees(angle_rad)

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
            
            # Ensure columns exist
            for col in [col_d, col_m, col_p]:
                if col not in df.columns:
                    raise ValueError(f"Column '{col}' not found in the dataframe.")

            # Define orthogonal vectors
            signal_v = df[col_p] - df[col_m]
            signal_h = df[col_d] - df[col_p]

            # Calculate axis using both methods
            axis_integral = calculate_axis_integral(signal_h, signal_v)
            axis_amplitude = calculate_axis_amplitude(signal_h, signal_v)
            
            # Add new columns
            df['cardiac_axis_integral'] = axis_integral
            df['cardiac_axis_amplitude'] = axis_amplitude
            
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
    
    # The script is run from the project root, so the relative path is correct.
    process_subject(args.subject_dir, args.col_d, args.col_m, args.col_p)

if __name__ == "__main__":
    main()
