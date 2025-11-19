import os
import sys
import argparse
import pandas as pd
import numpy as np
from glob import glob
from tqdm import tqdm
import neurokit2 as nk

# Add base directory to sys.path
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(base_dir)
from config.settings import (
    DATA_DIR,
    BASE_DIR,
    PROCESSED_DATA_DIR,
    OUTPUT_DIR,
    RAW_DATA_DIR,
    TEST_DIR,
    RATE,
    RATE_16CH,
    TIME,
    DATASET_MADE_DATE,
)
from config.name_dic import select_name_and_date


def find_qrs_peaks_fallback(
    signal, r_peak_center=150, r_search_window=5, qs_search_offset=50
):
    """
    (Fallback) Finds Q, R, S peak indices in a given signal, accounting for polarity inversion.
    """
    signal_np = signal.to_numpy() if isinstance(signal, pd.Series) else signal
    signal_len = len(signal_np)

    # Determine polarity by finding the point with max absolute value around the center
    center_search_start = r_peak_center - r_search_window
    center_search_end = r_peak_center + r_search_window
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


def find_qrs_peaks(signal, sampling_rate, r_peak_center=150):
    """
    Finds Q, R, S peak indices in a given signal using neurokit2, with a fallback.
    """
    signal_np = signal.to_numpy() if isinstance(signal, pd.Series) else signal

    try:
        # Delineate the ECG signal to find P, Q, R, S, T waves
        _, waves = nk.ecg_delineate(
            signal_np, sampling_rate=sampling_rate, method="dwt"
        )

        q_peaks = waves.get("ECG_Q_Peaks", [])
        r_peaks = waves.get("ECG_R_Peaks", [])
        s_peaks = waves.get("ECG_S_Peaks", [])

        # If no R peak is found, it's a failure.
        if not r_peaks:
            raise ValueError("No R-peak found by neurokit.")

        # Use the first detected peak in the single-beat waveform
        q_peak_idx = q_peaks[0] if q_peaks else r_peaks[0]
        r_peak_idx = r_peaks[0]
        s_peak_idx = s_peaks[0] if s_peaks else r_peaks[0]

        # Sanity check indices
        sig_len = len(signal_np)
        if not (
            0 <= q_peak_idx < sig_len
            and 0 <= r_peak_idx < sig_len
            and 0 <= s_peak_idx < sig_len
        ):
            raise ValueError("Peak index out of bounds.")

        return q_peak_idx, r_peak_idx, s_peak_idx

    except Exception:
        # If neurokit fails, use the original fallback method
        return find_qrs_peaks_fallback(signal, r_peak_center=r_peak_center)


def calculate_axis_from_12lead(signal_I, signal_aVF):
    """Calculates cardiac axis from Lead I and aVF signals based on net QRS amplitude."""
    # Find QRS peaks for each lead
    q_peak_idx_I, r_peak_idx_I, s_peak_idx_I = find_qrs_peaks(
        signal_I, sampling_rate=500
    )
    q_peak_idx_aVF, r_peak_idx_aVF, s_peak_idx_aVF = find_qrs_peaks(
        signal_aVF, sampling_rate=500
    )

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

    return {
        "axis_degrees": np.degrees(angle_rad),
        "q_idx_I": q_peak_idx_I,
        "r_idx_I": r_peak_idx_I,
        "s_idx_I": s_peak_idx_I,
        "q_amp_I": q_amp_I,
        "r_amp_I": r_amp_I,
        "s_amp_I": s_amp_I,
        "q_idx_aVF": q_peak_idx_aVF,
        "r_idx_aVF": r_peak_idx_aVF,
        "s_idx_aVF": s_peak_idx_aVF,
        "q_amp_aVF": q_amp_aVF,
        "r_amp_aVF": r_amp_aVF,
        "s_amp_aVF": s_amp_aVF,
    }


def get_physical_sensor_coordinates(orientation="normal"):
    """
    Returns the 2D coordinates for the 16 sensors.
    - 'normal': ch_1 is at the bottom-right (3,0).
    - 'flipped': ch_1 is at the top-left (0,3).
    """
    coords = {}
    for i in range(16):  # i is 0-indexed
        if orientation == "normal":
            # ch_1 is at (3,0)
            col = 3 - (i // 4)
            row = i % 4
        elif orientation == "flipped":
            # ch_1 is at (0,3)
            col = i // 4
            row = 3 - (i % 4)
        else:
            raise ValueError(
                "Invalid orientation specified. Use 'normal' or 'flipped'."
            )

        # coords key is 1-indexed
        coords[f"ch_{i+1}"] = (col, row)
    return coords


def create_virtual_electrodes(df_16ch, orientation="normal"):
    """
    Creates a 3x3 grid of 9 virtual electrodes by averaging 2x2 blocks of real electrodes.
    """
    physical_coords = get_physical_sensor_coordinates(orientation)
    # Create a reverse map from coordinate to channel name
    coord_to_ch = {v: k for k, v in physical_coords.items()}

    virtual_df = pd.DataFrame(index=df_16ch.index)
    virtual_coords = {}

    v_idx = 0
    # Iterate through the 3x3 possible virtual electrode positions
    for y_base in range(2, -1, -1):  # y from 2.5 down to 0.5 -> grid row 0, 1, 2
        for x_base in range(3):  # x from 0.5 to 2.5 -> grid col 0, 1, 2
            v_idx += 1
            v_name = f"v_ch_{v_idx}"

            # Define the 2x2 block of physical channels
            # Top-left corner of the block is (x_base, y_base + 1)
            ch_tl = coord_to_ch.get((x_base, y_base + 1))
            ch_tr = coord_to_ch.get((x_base + 1, y_base + 1))
            ch_bl = coord_to_ch.get((x_base, y_base))
            ch_br = coord_to_ch.get((x_base + 1, y_base))

            channels_to_average = [
                ch for ch in [ch_tl, ch_tr, ch_bl, ch_br] if ch is not None
            ]

            # Calculate the average waveform
            virtual_df[v_name] = df_16ch[channels_to_average].mean(axis=1)

            # Assign new centered coordinates for the virtual grid
            # New origin is at the center of the 3x3 grid (1, 1)
            virtual_coords[v_name] = (x_base - 1, (2 - y_base) - 1)

    return virtual_df, virtual_coords


def calculate_axis_from_virtual_electrodes(df_virtual_9ch):
    """
    Calculates cardiac axis from the 9 virtual electrodes using the new vector definition.
    H: v_ch_6 - v_ch_5 (Rightward)
    V: v_ch_8 - v_ch_5 (Downward)
    """
    signal_h = df_virtual_9ch["v_ch_6"] - df_virtual_9ch["v_ch_5"]
    signal_v = df_virtual_9ch["v_ch_8"] - df_virtual_9ch["v_ch_5"]

    # Use the re-peaking method for accurate amplitude calculation
    q_peak_idx_h, r_peak_idx_h, s_peak_idx_h = find_qrs_peaks(
        signal_h, sampling_rate=500
    )
    q_peak_idx_v, r_peak_idx_v, s_peak_idx_v = find_qrs_peaks(
        signal_v, sampling_rate=500
    )

    signal_h_np = signal_h.to_numpy()
    signal_v_np = signal_v.to_numpy()

    # Get amplitudes
    q_amp_h = signal_h_np[q_peak_idx_h]
    r_amp_h = signal_h_np[r_peak_idx_h]
    s_amp_h = signal_h_np[s_peak_idx_h]

    q_amp_v = signal_v_np[q_peak_idx_v]
    r_amp_v = signal_v_np[r_peak_idx_v]
    s_amp_v = signal_v_np[s_peak_idx_v]

    # Calculate net QRS amplitude
    net_amp_h = q_amp_h + r_amp_h + s_amp_h
    net_amp_v = q_amp_v + r_amp_v + s_amp_v

    angle_rad = np.arctan2(net_amp_v, net_amp_h)

    return {
        "axis_degrees": np.degrees(angle_rad),
        "q_idx_h": q_peak_idx_h,
        "r_idx_h": r_peak_idx_h,
        "s_idx_h": s_peak_idx_h,
        "q_amp_h": q_amp_h,
        "r_amp_h": r_amp_h,
        "s_amp_h": s_amp_h,
        "q_idx_v": q_peak_idx_v,
        "r_idx_v": r_peak_idx_v,
        "s_idx_v": s_peak_idx_v,
        "q_amp_v": q_amp_v,
        "r_amp_v": r_amp_v,
        "s_amp_v": s_amp_v,
    }


# --- Main Logic ---


def main(args):
    # The base directory is the project root. We use this to build robust paths.
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )

    # Setup paths
    subject_path = os.path.normpath(args.subject_dir)
    subject_name_full = os.path.basename(subject_path)
    subject_name_short = subject_name_full.split("_")[0]

    # Define the new dataset directory for virtual electrodes and axis
    output_subject_dir = f"{subject_name_full}_{args.orientation}"
    output_dataset_dir = os.path.join(
        project_root,
        "data",
        "processed",
        "virtual_electrode_dataset",
        output_subject_dir,
    )
    create_directory_if_not_exists(output_dataset_dir)

    # Find all dataset files to process
    data_dir = os.path.join(args.subject_dir, "0")
    files_to_process = sorted(glob(os.path.join(data_dir, "dataset_*.csv")))
    if not files_to_process:
        print(f"No dataset files found in {data_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Averaging {len(files_to_process)} beats for subject: {subject_name_full}")

    try:
        # --- Load all beats and calculate the average waveform ---
        all_beats = []
        for file_path in tqdm(
            files_to_process, desc=f"Loading beats for {subject_name_full}"
        ):
            df = pd.read_csv(file_path)
            all_beats.append(df)

        mean_waveform_df = pd.concat(all_beats).groupby(level=0).mean()

        # --- Extract 16-channel and 12-lead data ---
        ch_16_names = [f"ch_{i+1}" for i in range(16)]
        ch_12_names = [
            col
            for col in mean_waveform_df.columns
            if col not in ch_16_names and "Time" not in col
        ]

        df_16ch = mean_waveform_df[ch_16_names]
        df_12lead = mean_waveform_df[ch_12_names]

        # --- Create virtual electrodes ---
        print("Creating 9-channel virtual electrode dataset...")
        df_virtual_9ch, _ = create_virtual_electrodes(df_16ch, args.orientation)

        # --- Save the mean virtual waveform dataset ---
        if "Time" in mean_waveform_df.columns:
            df_virtual_9ch_to_save = pd.concat(
                [mean_waveform_df["Time"], df_virtual_9ch, df_12lead], axis=1
            )
        else:
            df_virtual_9ch_to_save = pd.concat([df_virtual_9ch, df_12lead], axis=1)

        virtual_csv_path = os.path.join(output_dataset_dir, "mean_virtual_waveform.csv")
        df_virtual_9ch_to_save.to_csv(
            virtual_csv_path, index=False, float_format="%.6f"
        )
        print(f"Mean virtual waveform data saved to {virtual_csv_path}")

        # --- Calculate axis using virtual electrodes ---
        print("Calculating axis from virtual electrodes...")
        virtual_axis_data = calculate_axis_from_virtual_electrodes(df_virtual_9ch)

        # --- Calculate axis from 12-lead ECG for comparison ---
        print("Calculating axis from 12-lead ECG...")
        lead12_axis_data = calculate_axis_from_12lead(
            df_12lead[args.col_I], df_12lead[args.col_aVF]
        )

        # --- Save the axis comparison results ---
        # Combine all data into a single dictionary for DataFrame creation
        output_data = {
            "subject": [subject_name_short],
            "cardiac_axis_virtual_degrees": [virtual_axis_data["axis_degrees"]],
            "cardiac_axis_12lead_degrees": [lead12_axis_data["axis_degrees"]],
            # Add virtual electrode QRS data
            "virtual_q_idx_h": [virtual_axis_data["q_idx_h"]],
            "virtual_r_idx_h": [virtual_axis_data["r_idx_h"]],
            "virtual_s_idx_h": [virtual_axis_data["s_idx_h"]],
            "virtual_q_amp_h": [virtual_axis_data["q_amp_h"]],
            "virtual_r_amp_h": [virtual_axis_data["r_amp_h"]],
            "virtual_s_amp_h": [virtual_axis_data["s_amp_h"]],
            "virtual_q_idx_v": [virtual_axis_data["q_idx_v"]],
            "virtual_r_idx_v": [virtual_axis_data["r_idx_v"]],
            "virtual_s_idx_v": [virtual_axis_data["s_idx_v"]],
            "virtual_q_amp_v": [virtual_axis_data["q_amp_v"]],
            "virtual_r_amp_v": [virtual_axis_data["r_amp_v"]],
            "virtual_s_amp_v": [virtual_axis_data["s_amp_v"]],
        }
        axis_df = pd.DataFrame(output_data)

        axis_csv_path = os.path.join(output_dataset_dir, "virtual_axis.csv")
        axis_df.to_csv(axis_csv_path, index=False, float_format="%.4f")

        print(f"\nAxis comparison saved to {axis_csv_path}")
        print(
            f"  - Virtual Electrode Axis: {virtual_axis_data['axis_degrees']:.2f} degrees"
        )
        print(f"  - 12-Lead Axis: {lead12_axis_data['axis_degrees']:.2f} degrees")

    except Exception as e:
        print(f"An error occurred during processing: {e}", file=sys.stderr)


def create_directory_if_not_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)


if __name__ == "__main__":
    # Arguments are set directly for simplicity
    class Args:
        pass

    args = Args()

    # ##################################################################
    # #############  CONFIGURATION FOR THE SCRIPT RUN  ###############
    # ##################################################################
    subject_name_full = "noda_0714_0.8s"
    args.orientation = "flipped"  # <--- CHANGE THIS: "normal" or "flipped"
    # ##################################################################

    # The base directory is the project root. We use this to build robust paths.
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )

    args.subject_dir = os.path.join(
        project_root, "data", "processed", "for_best_resample", subject_name_full
    )
    args.col_I = "A1"
    args.col_aVF = "aVF"
    main(args)
