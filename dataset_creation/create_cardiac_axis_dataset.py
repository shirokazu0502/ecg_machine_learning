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

    signal_h_np = signal_h.to_numpy()
    signal_v_np = signal_v.to_numpy()

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


def calculate_axis_from_12lead(signal_I, signal_aVF):
    """Calculates cardiac axis from Lead I and aVF signals based on net QRS amplitude."""
    # Find QRS peaks for each lead
    q_peak_idx_I, r_peak_idx_I, s_peak_idx_I = find_qrs_peaks(signal_I)
    q_peak_idx_aVF, r_peak_idx_aVF, s_peak_idx_aVF = find_qrs_peaks(signal_aVF)

    # Ensure signals are numpy arrays for indexing
    signal_I_np = signal_I.to_numpy() if isinstance(signal_I, pd.Series) else signal_I
    signal_aVF_np = (
        signal_aVF.to_numpy() if isinstance(signal_aVF, pd.Series) else signal_aVF
    )

    # Get amplitudes
    q_amp_I = signal_I_np[q_peak_idx_I]
    r_amp_I = signal_I_np[r_peak_idx_I]
    s_amp_I = signal_I_np[s_peak_idx_I]

    q_amp_aVF = signal_aVF_np[q_peak_idx_aVF]
    r_amp_aVF = signal_aVF_np[r_peak_idx_aVF]
    s_amp_aVF = signal_aVF_np[s_peak_idx_aVF]

    # Calculate net QRS amplitude (R - (Q+S))
    net_amp_I = r_amp_I + q_amp_I + s_amp_I
    net_amp_aVF = r_amp_aVF + q_amp_aVF + s_amp_aVF

    # Calculate angle using arctan2(y, x)
    angle_rad = np.arctan2(net_amp_aVF, net_amp_I)
    return np.degrees(angle_rad)


def process_subject(subject_dir, col_base, col_h, col_v, col_I, col_aVF):
    """
    Processes all heartbeat CSVs for a single subject to calculate and save the mean cardiac axis.
    """
    source_data_path = os.path.join(subject_dir, "0", "moving_ave_datasets")
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

        # --- 15-channel sensor method (new definition) ---
        required_cols_15ch = [col_base, col_h, col_v]
        if not all(col in mean_waveform_df.columns for col in required_cols_15ch):
            raise ValueError(
                f"One or more columns for 15-ch axis calculation not found."
            )
        signal_h = (
            mean_waveform_df[col_h] - mean_waveform_df[col_base]
        )  # Rightward vector
        signal_v = (
            mean_waveform_df[col_v] - mean_waveform_df[col_base]
        )  # Downward vector
        axis_15ch = calculate_axis_with_re_peaking(signal_h, signal_v)

        # --- 12-lead ECG method ---
        required_cols_12lead = [col_I, col_aVF]
        if not all(col in mean_waveform_df.columns for col in required_cols_12lead):
            raise ValueError(
                f"Columns '{col_I}' or '{col_aVF}' for 12-lead axis calculation not found."
            )
        axis_12lead = calculate_axis_from_12lead(
            mean_waveform_df[col_I], mean_waveform_df[col_aVF]
        )

        # --- Save the results ---
        subject_short_name = subject_name.split("_")[0]
        axis_df = pd.DataFrame(
            {
                "subject": [subject_short_name],
                "cardiac_axis_15ch_degrees": [axis_15ch],
                "cardiac_axis_12lead_degrees": [axis_12lead],
            }
        )
        axis_csv_path = os.path.join(
            output_dir, f"{subject_short_name}_cardiac_axis.csv"
        )
        axis_df.to_csv(axis_csv_path, index=False, float_format="%.2f")
        print(f"Cardiac axis comparison saved to {axis_csv_path}")

        # 2. Save the mean waveform data to a CSV file
        mean_waveform_csv_path = os.path.join(output_dir, "mean_waveform.csv")
        mean_waveform_df.to_csv(mean_waveform_csv_path, index=False)
        print(f"Mean waveform saved to {mean_waveform_csv_path}")
        print(f"Finished processing for {subject_name}.")
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
    # Argument for the source data directory
    parser.add_argument(
        "--subject_dir",
        type=str,
        default="data/processed/15ch_arrange_direction/asano_0714_0.8s",
        help="Path to the subject's data directory, relative to the project root.",
    )
    # Arguments for the new 15-channel vector definition
    parser.add_argument(
        "--col_base",
        type=str,
        default="ch_5",
        help="Column name for the base/origin channel (default: 'ch_5').",
    )
    parser.add_argument(
        "--col_h",
        type=str,
        default="ch_13",
        help="Column name for the horizontal vector endpoint (default: 'ch_13').",
    )
    parser.add_argument(
        "--col_v",
        type=str,
        default="ch_8",
        help="Column name for the vertical vector endpoint (default: 'ch_8').",
    )
    # Arguments for 12-lead ECG data
    parser.add_argument(
        "--col_I",
        type=str,
        default="A1",
        help="Column name for Lead I (default: 'A1').",
    )
    parser.add_argument(
        "--col_aVF",
        type=str,
        default="aVF",
        help="Column name for Lead aVF (default: 'aVF').",
    )
    args = parser.parse_args()
    # Construct the full, absolute path for subject_dir from the project root.
    full_subject_path = os.path.join(project_root, args.subject_dir)
    process_subject(
        full_subject_path,
        args.col_base,
        args.col_h,
        args.col_v,
        args.col_I,
        args.col_aVF,
    )


if __name__ == "__main__":
    main()
