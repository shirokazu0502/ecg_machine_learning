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

def calculate_axis_integral(signal_h, signal_v):
    """Calculates cardiac axis based on signal area (integral)."""
    qrs_integral_v = np.sum(signal_v[QRS_WINDOW_START:QRS_WINDOW_END])
    qrs_integral_h = np.sum(signal_h[QRS_WINDOW_START:QRS_WINDOW_END])
    angle_rad = np.arctan2(qrs_integral_v, qrs_integral_h)
    return np.degrees(angle_rad)

def calculate_axis_amplitude_from_peaks(signal_h, signal_v, q_peak_idx, r_peak_idx, s_peak_idx):
    """Calculates cardiac axis based on QRS amplitude using provided peak indices."""
    
    signal_len = len(signal_h)
    if not all(0 <= idx < signal_len for idx in [q_peak_idx, r_peak_idx, s_peak_idx]):
        raise IndexError(f"Peak index out of bounds. Indices: [Q:{q_peak_idx}, R:{r_peak_idx}, S:{s_peak_idx}], Signal length: {signal_len}")

    # Convert to numpy for consistent indexing if they are pandas Series
    signal_h_np = signal_h.to_numpy()
    signal_v_np = signal_v.to_numpy()

    # Extract amplitudes directly using indices, respecting their signs
    q_amp_h = signal_h_np[q_peak_idx]
    r_amp_h = signal_h_np[r_peak_idx]
    s_amp_h = signal_h_np[s_peak_idx]
    
    q_amp_v = signal_v_np[q_peak_idx]
    r_amp_v = signal_v_np[r_peak_idx]
    s_amp_v = signal_v_np[s_peak_idx]

    # Net amplitude is the algebraic sum of Q, R, and S amplitudes.
    net_amp_h = q_amp_h + r_amp_h + s_amp_h
    net_amp_v = q_amp_v + r_amp_v + s_amp_v

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

    # Define paths
    subject_name = os.path.basename(subject_dir)
    output_dir = os.path.join(base_dir, "data", "processed", "cardiac_axis_dataset", subject_name, "0", "moving_ave_datasets")
    os.makedirs(output_dir, exist_ok=True)
    ponset_dir = os.path.join(subject_dir, "0") # Path to ponset_toffset files

    # Find all dataset files
    files_to_process = sorted(glob.glob(os.path.join(source_data_path, "dataset_*.csv")))
    if not files_to_process:
        print(f"No dataset CSVs found for subject {subject_name}")
        return

    print(f"Processing {len(files_to_process)} files for subject: {subject_name}")
    
    for file_path in tqdm(files_to_process, desc=f"Processing {subject_name}"):
        try:
            # --- Load beat signal data ---
            df = pd.read_csv(file_path)
            
            # --- Load corresponding peak index data ---
            file_id = os.path.basename(file_path).split('_')[-1].replace('.csv', '')
            ponset_path = os.path.join(ponset_dir, f"ponset_toffset_{file_id}.csv")
            
            if not os.path.exists(ponset_path):
                print(f"Warning: ponset_toffset file not found for {os.path.basename(file_path)}. Cardiac axis (amplitude) will not be calculated.")
                df['cardiac_axis_amplitude'] = np.nan # Add column with NaN
            else:
                peaks_df = pd.read_csv(ponset_path)
                # Get indices from the first row of the peaks file
                peak_indices = peaks_df.iloc[0]
                q_peak_idx = int(peak_indices['q_peak'])
                r_peak_idx = 150 # Hardcoded value as requested
                s_peak_idx = int(peak_indices['s_peak'])

                # Ensure columns exist
                for col in [col_d, col_m, col_p]:
                    if col not in df.columns:
                        raise ValueError(f"Column '{col}' not found in the dataframe.")

                # Define orthogonal vectors
                signal_v = df[col_p] - df[col_m]
                signal_h = df[col_d] - df[col_p]

                # Calculate axis using the new amplitude method
                axis_amplitude = calculate_axis_amplitude_from_peaks(signal_h, signal_v, q_peak_idx, r_peak_idx, s_peak_idx)
                df['cardiac_axis_amplitude'] = axis_amplitude

            # --- Calculate integral-based axis (unchanged) ---
            # This part is kept as is, assuming it's still needed for comparison.
            # Ensure columns exist for this calculation as well
            required_cols_integral = [col_d, col_m, col_p]
            if all(col in df.columns for col in required_cols_integral):
                signal_v_integral = df[col_p] - df[col_m]
                signal_h_integral = df[col_d] - df[col_p]
                axis_integral = calculate_axis_integral(signal_h_integral, signal_v_integral)
                df['cardiac_axis_integral'] = axis_integral
            else:
                df['cardiac_axis_integral'] = np.nan


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
    
    process_subject(args.subject_dir, args.col_d, args.col_m, args.col_p)

if __name__ == "__main__":
    main()