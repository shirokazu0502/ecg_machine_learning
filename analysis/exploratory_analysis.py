import pandas as pd
import numpy as np
import os
from glob import glob
import matplotlib.pyplot as plt


def analyze_waveform_properties():
    """
    Analyzes and compares waveform properties (peak amplitudes, interval durations)
    between 15-channel data (ch_1) and medical ECG data (A2) for each subject.
    """
    # --- Configuration ---
    BASE_DIR = "/mnt/ecg_project/data/processed/15ch_arrange_direction"
    OUTPUT_DIR = "analysis/output"
    SUBJECT_DIRS = [d for d in glob(f"{BASE_DIR}/*/") if os.path.isdir(d)]

    # To store features from all heartbeats across all subjects
    all_features = []

    print(f"Found {len(SUBJECT_DIRS)} subjects. Starting analysis...")

    # --- Data Loading and Feature Extraction ---
    for subject_path in SUBJECT_DIRS:
        subject_name = os.path.basename(os.path.normpath(subject_path))
        print(f"  Processing subject: {subject_name}...")

        # Using the '0' subdirectory as the source of heartbeats
        heartbeat_dir = os.path.join(subject_path, "0")
        dataset_files = sorted(glob(os.path.join(heartbeat_dir, "dataset_*.csv")))

        if not dataset_files:
            print(f"    - No dataset files found for {subject_name}. Skipping.")
            continue

        for i, dataset_file in enumerate(dataset_files):
            ponset_file = dataset_file.replace("dataset_", "ponset_toffset_")

            if not os.path.exists(ponset_file):
                continue

            try:
                df_data = pd.read_csv(dataset_file)
                df_pt = pd.read_csv(ponset_file)
                pt_array = df_pt.iloc[0].to_dict()

                # --- Extract Features for one heartbeat ---
                features = {"subject": subject_name, "heartbeat_id": i}

                # Interval Durations (calculated once from pt_array, in milliseconds)
                # Sampling rate is 500 Hz, so 1 index = 2 ms
                sampling_rate = 500
                ms_per_index = 1000 / sampling_rate

                features["pr_interval_ms"] = (
                    pt_array["q_peak"] - pt_array["p_onset"]
                ) * ms_per_index
                features["qrs_duration_ms"] = (
                    pt_array["s_peak"] - pt_array["q_peak"]
                ) * ms_per_index
                features["qt_interval_ms"] = (
                    pt_array["t_offset"] - pt_array["q_peak"]
                ) * ms_per_index

                # Peak Amplitudes
                peak_keys = ["p_peak", "q_peak", "r_peak", "s_peak", "t_peak"]
                for peak_key in peak_keys:
                    peak_idx = int(pt_array[peak_key])

                    # Ensure index is within bounds
                    if 0 <= peak_idx < len(df_data):
                        # For medical ECG ('A2')
                        features[f"amp_A2_{peak_key}"] = df_data.loc[peak_idx, "A2"]

                        # For 15-ch data ('ch_1' or fallback to 'ch_16')
                        ch_to_use = None
                        if "ch_1" in df_data.columns:
                            ch_to_use = "ch_1"
                        elif "ch_16" in df_data.columns:
                            ch_to_use = "ch_16"

                        if ch_to_use:
                            features[f"amp_ch1_{peak_key}"] = df_data.loc[
                                peak_idx, ch_to_use
                            ]
                        else:
                            features[f"amp_ch1_{peak_key}"] = np.nan
                            print(
                                f"      WARNING: Neither 'ch_1' nor 'ch_16' found for {subject_name} heartbeat {i}. Amplitude for 15ch data set to NaN."
                            )
                    else:
                        for ch_key in ["A2", "ch1"]:
                            features[f"amp_{ch_key}_{peak_key}"] = np.nan

                all_features.append(features)

            except Exception as e:
                print(
                    f"    - Could not process file {os.path.basename(dataset_file)}. Error: {e}"
                )

    if not all_features:
        print("No features were extracted. Stopping analysis.")
        return

    df_features = pd.DataFrame(all_features)

    # --- Per-Subject Statistical Analysis ---
    # Calculate mean and std for each feature per subject
    subject_summary = df_features.groupby("subject").agg(["mean", "std"])

    # Clean up column names for better readability
    subject_summary.columns = [
        "_".join(col).strip() for col in subject_summary.columns.values
    ]

    # Save summary to CSV
    summary_path = os.path.join(OUTPUT_DIR, "subject_summary_statistics.csv")
    subject_summary.to_csv(summary_path)
    print(f"\nSaved per-subject statistical summary to: {summary_path}")

    # --- Visualization ---
    print("Generating visualizations...")

    # Select key features to plot
    features_to_plot = [
        "pr_interval_ms",
        "qrs_duration_ms",
        "qt_interval_ms",
        "amp_A2_r_peak",
        "amp_ch1_r_peak",
        "amp_A2_p_peak",
        "amp_ch1_p_peak",
        "amp_A2_t_peak",
        "amp_ch1_t_peak",
    ]

    for feature in features_to_plot:
        plt.figure(figsize=(15, 8))
        df_features.boxplot(column=feature, by="subject", grid=True, rot=45)
        plt.title(f"Distribution of {feature} by Subject")
        plt.ylabel(feature)
        plt.xlabel("Subject")
        plt.tight_layout()

        plot_path = os.path.join(OUTPUT_DIR, f"boxplot_{feature}.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"  - Saved box plot to: {plot_path}")

    print("\nAnalysis complete.")


if __name__ == "__main__":
    analyze_waveform_properties()
