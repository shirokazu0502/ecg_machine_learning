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


def find_qrs_peaks(signal, r_peak_center=150, r_search_window=25, qs_search_offset=50):
    """
    Finds Q, R, S peak indices in a given signal, accounting for polarity inversion.
    """
    signal_np = signal.to_numpy() if isinstance(signal, pd.Series) else signal
    signal_len = len(signal_np)

    # Determine polarity by finding the point with max absolute value around the center
    center_search_start = max(0, r_peak_center - r_search_window)
    center_search_end = min(signal_len, r_peak_center + r_search_window)
    center_slice = signal_np[center_search_start:center_search_end]

    if len(center_slice) == 0:  # Handle empty slice
        return r_peak_center, r_peak_center, r_peak_center

    abs_max_idx_in_slice = np.argmax(np.abs(center_slice))
    is_inverted = center_slice[abs_max_idx_in_slice] < 0

    # --- Find Peaks based on polarity ---
    if not is_inverted:  # Normal QRS (R is positive)
        r_peak_idx = center_search_start + np.argmax(center_slice)

        q_search_start = max(0, r_peak_idx - qs_search_offset)
        q_search_end = r_peak_idx
        q_peak_idx = (
            q_search_start + np.argmin(signal_np[q_search_start:q_search_end])
            if q_search_start < q_search_end
            else r_peak_idx
        )

        s_search_start = r_peak_idx
        s_search_end = min(signal_len, r_peak_idx + qs_search_offset)
        s_peak_idx = (
            s_search_start + np.argmin(signal_np[s_search_start:s_search_end])
            if s_search_start < s_search_end
            else r_peak_idx
        )
    else:  # Inverted QRS (R is negative)
        r_peak_idx = center_search_start + np.argmin(center_slice)

        q_search_start = max(0, r_peak_idx - qs_search_offset)
        q_search_end = r_peak_idx
        q_peak_idx = (
            q_search_start + np.argmax(signal_np[q_search_start:q_search_end])
            if q_search_start < q_search_end
            else r_peak_idx
        )

        s_search_start = r_peak_idx
        s_search_end = min(signal_len, r_peak_idx + qs_search_offset)
        s_peak_idx = (
            s_search_start + np.argmax(signal_np[s_search_start:s_search_end])
            if s_search_start < s_search_end
            else r_peak_idx
        )

    return q_peak_idx, r_peak_idx, s_peak_idx


def calculate_axis_with_re_peaking(signal_h, signal_v):
    """
    Calculates cardiac axis by re-finding QRS peaks on the orthogonal signals.
    """
    q_peak_idx_h, r_peak_idx_h, s_peak_idx_h = find_qrs_peaks(signal_h)
    q_peak_idx_v, r_peak_idx_v, s_peak_idx_v = find_qrs_peaks(signal_v)

    signal_h_np = signal_h.to_numpy() if isinstance(signal, pd.Series) else signal
    signal_v_np = signal_v.to_numpy() if isinstance(signal, pd.Series) else signal

    # Extract amplitudes using re-found peaks
    q_amp_h = signal_h_np[q_peak_idx_h]
    r_amp_h = signal_h_np[r_peak_idx_h]
    s_amp_h = signal_h_np[s_peak_idx_h]

    q_amp_v = signal_v_np[q_peak_idx_v]
    r_amp_v = signal_v_np[r_peak_idx_v]
    s_amp_v = signal_v_np[s_peak_idx_v]

    # Net amplitude is the algebraic sum of Q, R, and S amplitudes.
    net_amp_h = q_amp_h + r_amp_h + s_amp_h
    net_amp_v = q_amp_v + r_amp_v + s_amp_v

    angle_rad = np.arctan2(net_amp_v, net_amp_h)
    return np.degrees(angle_rad)


def process_subject(subject_dir, col_d, col_m, col_p):
    """
    Processes all heartbeat CSVs for a single subject to calculate and save the mean cardiac axis.
    """
    source_data_path = os.path.join(subject_dir, "moving_ave_datasets")
    if not os.path.isdir(source_data_path):
        print(f"Error: 'moving_ave_datasets' directory not found in {subject_dir}")
        return

    # Define paths
    subject_name = os.path.basename(subject_dir)
    output_dir = os.path.join(
        base_dir, "data", "processed", "cardiac_axis_dataset", subject_name, "0"
    )
    os.makedirs(output_dir, exist_ok=True)

    # Find all dataset files
    files_to_process = sorted(
        glob.glob(os.path.join(source_data_path, "dataset_*.csv"))
    )
    if not files_to_process:
        print(f"No dataset CSVs found for subject {subject_name}")
        return

    print(f"Averaging {len(files_to_process)} beats for subject: {subject_name}")

    try:
        # --- Load all beats and calculate the average waveform ---
        all_beats = []
        # Use the first file to get header information
        first_df = pd.read_csv(files_to_process[0])
        column_names = first_df.columns

        for file_path in tqdm(
            files_to_process, desc=f"Loading beats for {subject_name}"
        ):
            df = pd.read_csv(file_path)
            all_beats.append(df.to_numpy())

        # Calculate the mean across all beats
        mean_waveform_np = np.mean(all_beats, axis=0)
        mean_waveform_df = pd.DataFrame(mean_waveform_np, columns=column_names)

        # --- Calculate cardiac axis from the average waveform ---
        print("Calculating cardiac axis from average waveform...")
        # Ensure columns exist
        for col in [col_d, col_m, col_p]:
            if col not in mean_waveform_df.columns:
                raise ValueError(
                    f"Column '{col}' not found in the average waveform dataframe."
                )

        # Define orthogonal vectors
        signal_v = mean_waveform_df[col_p] - mean_waveform_df[col_m]
        signal_h = mean_waveform_df[col_d] - mean_waveform_df[col_p]

        # Calculate axis using the new amplitude method
        axis_amplitude = calculate_axis_with_re_peaking(signal_h, signal_v)

        # --- Save the result to a summary file ---
        summary_filepath = os.path.join(output_dir, "cardiac_axis_summary.txt")
        with open(summary_filepath, "w") as f:
            f.write(f"# Cardiac Axis Calculation Summary\n")
            f.write(f"Subject: {subject_name}\n")
            f.write(f"Number of beats averaged: {len(all_beats)}\n")
            f.write(
                f"Calculated cardiac axis (amplitude-based): {axis_amplitude:.2f} degrees\n"
            )

        print(
            f"Finished processing for {subject_name}. Summary saved in {summary_filepath}"
        )

    except Exception as e:
        print(f"Could not process subject {subject_name}. Error: {e}")


def main():

    parser = argparse.ArgumentParser(
        description="Create a new dataset with cardiac axis information."
    )

    # The base directory is the project root. We use this to build robust paths.

    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )

    parser.add_argument(
        "--subject_dir",
        type=str,
        default="data/processed/15ch_arrange_direction/asano_0714_0.8s",
        help="Path to the subject's data directory, relative to the project root.",
    )

    parser.add_argument(
        "--col_d",
        type=str,
        default="ch_4",
        help="Column name for the Lower Left position (default: 'ch_4').",
    )

    parser.add_argument(
        "--col_m",
        type=str,
        default="ch_13",
        help="Column name for the Upper Right position (default: 'ch_13').",
    )

    parser.add_argument(
        "--col_p",
        type=str,
        default="ch_16",
        help="Column name for the Lower Right position (default: 'ch_16').",
    )

    args = parser.parse_args()

    # Construct the full, absolute path for subject_dir from the project root.

    # This resolves issues with relative paths like '../../' and makes execution location independent.

    full_subject_path = os.path.join(project_root, args.subject_dir)

    process_subject(full_subject_path, args.col_d, args.col_m, args.col_p)


if __name__ == "__main__":

    main()
