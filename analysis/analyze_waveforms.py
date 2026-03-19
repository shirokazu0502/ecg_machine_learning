import os
import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import re
import math

# --- Configuration ---
DATASET_NAME = "reference_center_4ch"
BASE_DATA_DIR = "/mnt/ecg_project/data/processed"
OUTPUT_DIR = "output"
NUM_16CH_CHANNELS = 16  # Number of ch_1 to ch_16 channels
NUM_12LEAD_CHANNELS = 8  # Number of A1 to V6 channels
NUM_DATASETS_PER_SUBJECT = 200  # Max number of dataset_XXX.csv files to look for
DATA_LENGTH = 400

# Define channel names explicitly
ALL_16_CHANNELS_NAMES = [f"ch_{i+1}" for i in range(NUM_16CH_CHANNELS)]
MEDICAL_ECG_CHANNELS_NAMES = [
    "A1",
    "A2",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
]


def get_subject_dirs(directory_path):
    """Gets a list of subject directories."""
    subject_dirs = []
    for entry in os.scandir(directory_path):
        if entry.is_dir():
            subject_dirs.append(entry.name)
    return subject_dirs


def get_short_name(subject_dir_name):
    """Extracts a short name (e.g., 'asano') from the full directory name."""
    match = re.match(r"([a-zA-Z0-9_]+?_)", subject_dir_name)
    return match.group(1) if match else subject_dir_name


def run_analysis_for_base_ch(full_dataset_path, base_ch):
    """
    Runs the full analysis and plotting for a specific base channel directory (e.g., '0').
    """
    print(f"\n{'='*20}\nRunning analysis for: {base_ch}\n{'='*20}")

    subject_dirs = get_subject_dirs(full_dataset_path)
    if not subject_dirs:
        print(f"Error: No subject directories found in '{full_dataset_path}'")
        return

    all_subject_averages_16ch = {}
    all_subject_averages_12lead = {}

    # --- Data Accumulation for Averaging ---
    for subject_dir in subject_dirs:
        short_name = get_short_name(subject_dir)
        print(f"Processing subject: {short_name}...")

        subject_path = os.path.join(
            full_dataset_path, subject_dir, base_ch, "moving_ave_datasets"
        )

        if not os.path.isdir(subject_path):
            print(f"  - Directory not found, skipping: {subject_path}")
            continue

        accumulator_16ch = np.zeros((NUM_16CH_CHANNELS, DATA_LENGTH))
        count_16ch = 0
        accumulator_12lead = np.zeros((NUM_12LEAD_CHANNELS, DATA_LENGTH))
        count_12lead = 0

        for i in range(NUM_DATASETS_PER_SUBJECT):
            file_path = os.path.join(subject_path, f"dataset_{str(i).zfill(3)}.csv")

            if os.path.exists(file_path):
                try:
                    df = pd.read_csv(file_path, header=0)
                    df.columns = df.columns.str.strip()

                    # Process 16-channel data
                    if set(ALL_16_CHANNELS_NAMES).issubset(df.columns):
                        wave_data_16ch = df[ALL_16_CHANNELS_NAMES].T.values
                        if wave_data_16ch.shape == (NUM_16CH_CHANNELS, DATA_LENGTH):
                            accumulator_16ch += wave_data_16ch
                            count_16ch += 1
                        else:
                            print(
                                f"  - Skipping {os.path.basename(file_path)} 16ch data due to incorrect shape: {wave_data_16ch.shape}"
                            )

                    # Process 12-lead data
                    if set(MEDICAL_ECG_CHANNELS_NAMES).issubset(df.columns):
                        wave_data_12lead = df[MEDICAL_ECG_CHANNELS_NAMES].T.values
                        if wave_data_12lead.shape == (NUM_12LEAD_CHANNELS, DATA_LENGTH):
                            accumulator_12lead += wave_data_12lead
                            count_12lead += 1

                except Exception as e:
                    print(f"  - Error processing {file_path}: {e}")

        if count_16ch > 0:
            all_subject_averages_16ch[subject_dir] = accumulator_16ch / count_16ch
            print(f"  -> Calculated 16ch average over {count_16ch} heartbeats.")
        else:
            print(
                f"  -> No valid 16ch data found for subject {short_name} in {base_ch}."
            )

        if count_12lead > 0:
            all_subject_averages_12lead[subject_dir] = accumulator_12lead / count_12lead
            print(f"  -> Calculated 12lead average over {count_12lead} heartbeats.")
        else:
            print(
                f"  -> No valid 12lead data found for subject {short_name} in {base_ch}."
            )

    # --- Plotting Section ---
    if not all_subject_averages_16ch and not all_subject_averages_12lead:
        print(f"No data processed for {base_ch}. Skipping all plot generation.")
        return

    # Create base output directory for this base channel (e.g., 'output/0')
    base_ch_output_dir = os.path.join(OUTPUT_DIR, base_ch)
    os.makedirs(base_ch_output_dir, exist_ok=True)

    # --- Plot 1: Average Per-subject plots (16ch) ---
    if all_subject_averages_16ch:
        print(f"\nGenerating 16ch per-subject average plots for {base_ch}...")
        for subject_dir, avg_data in all_subject_averages_16ch.items():
            subject_name = os.path.basename(subject_dir)
            subject_output_path = os.path.join(base_ch_output_dir, subject_name)
            os.makedirs(subject_output_path, exist_ok=True)
            fig, axes = plt.subplots(4, 4, figsize=(20, 15))
            fig.suptitle(
                f"Average 16ch Waveforms: {subject_name} (Source: {base_ch})",
                fontsize=24,
            )
            axes = axes.flatten()
            for i in range(NUM_16CH_CHANNELS):
                axes[i].plot(avg_data[i, :])
                axes[i].set_title(
                    f"{ALL_16_CHANNELS_NAMES[i]}", fontsize=20, fontweight="bold"
                )
                axes[i].grid(True)
            for i in range(NUM_16CH_CHANNELS, len(axes)):
                axes[i].set_visible(False)
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            output_filename = os.path.join(subject_output_path, "subject_avg_16ch.png")
            plt.savefig(output_filename)
            plt.close(fig)
            print(f"  - Saved average plot: {output_filename}")

    # --- Plot 2: Average Per-subject plots (12-lead) ---
    if all_subject_averages_12lead:
        print(f"\nGenerating 12lead per-subject average plots for {base_ch}...")
        for subject_dir, avg_data in all_subject_averages_12lead.items():
            subject_name = os.path.basename(subject_dir)
            subject_output_path = os.path.join(base_ch_output_dir, subject_name)
            os.makedirs(subject_output_path, exist_ok=True)
            fig, axes = plt.subplots(3, 4, figsize=(20, 12))
            fig.suptitle(
                f"Average 12-Lead Waveforms: {subject_name} (Source: {base_ch})",
                fontsize=24,
            )
            axes = axes.flatten()
            for i in range(NUM_12LEAD_CHANNELS):
                axes[i].plot(avg_data[i, :])
                axes[i].set_title(
                    f"{MEDICAL_ECG_CHANNELS_NAMES[i]}", fontsize=20, fontweight="bold"
                )
                axes[i].grid(True)
            for i in range(NUM_12LEAD_CHANNELS, len(axes)):
                axes[i].set_visible(False)
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            output_filename = os.path.join(
                subject_output_path, "subject_avg_12lead.png"
            )
            plt.savefig(output_filename)
            plt.close(fig)
            print(f"  - Saved average plot: {output_filename}")

    # --- NEW: Plot 3: Individual Heartbeat Plots ---
    print(f"\nGenerating individual heartbeat plots for {base_ch}...")
    for subject_dir in subject_dirs:
        subject_path = os.path.join(full_dataset_path, subject_dir, base_ch)
        if os.path.isdir(subject_path):
            plot_individual_heartbeats(
                subject_path, os.path.basename(subject_dir), base_ch_output_dir
            )


def plot_individual_heartbeats(subject_path, subject_name, base_ch_output_dir):
    """
    Plots all individual heartbeats for a given subject path, combining 16ch and 12-lead data.
    """
    #
    # --- Comment: Start processing individual plots for a subject ---
    #
    print(f"  -> Plotting individual heartbeats for {subject_name}...")

    # Create a dedicated directory for these plots
    output_plot_dir = os.path.join(base_ch_output_dir, subject_name, "individual_plots")
    if os.path.exists(output_plot_dir):
        return  # Skip if already processed
    os.makedirs(output_plot_dir)

    # Find all individual dataset files
    dataset_files = sorted(glob.glob(os.path.join(subject_path, "dataset_*.csv")))

    if not dataset_files:
        print(f"    - No individual datasets found in {subject_path}")
        return

    #
    # --- Comment: Loop through each heartbeat file ---
    #
    for file_path in dataset_files:
        try:
            df = pd.read_csv(file_path, header=0)
            df.columns = df.columns.str.strip()

            # Identify available channels
            ch16_cols_present = [
                col for col in ALL_16_CHANNELS_NAMES if col in df.columns
            ]
            lead12_cols_present = [
                col for col in MEDICAL_ECG_CHANNELS_NAMES if col in df.columns
            ]

            total_channels = len(ch16_cols_present) + len(lead12_cols_present)
            if total_channels == 0:
                print(
                    f"    - No plottable channels found in {os.path.basename(file_path)}. Skipping."
                )
                continue

            #
            # --- Comment: Create a plot figure with a grid layout for all channels ---
            #
            ncols = 5
            nrows = math.ceil(total_channels / ncols)
            fig, axes = plt.subplots(
                nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False
            )
            fig.suptitle(
                f"Individual Heartbeat: {subject_name} - {os.path.basename(file_path)}",
                fontsize=16,
            )
            axes = axes.flatten()

            plot_idx = 0
            #
            # --- Comment: Plot 16-channel sensor data ---
            #
            for col in ch16_cols_present:
                ax = axes[plot_idx]
                ax.plot(df[col])
                ax.set_title(f"16ch: {col}", fontsize=16, fontweight="bold")
                ax.grid(True)
                plot_idx += 1

            #
            # --- Comment: Plot 12-lead standard ECG data ---
            #
            for col in lead12_cols_present:
                ax = axes[plot_idx]
                ax.plot(df[col], color="green")
                ax.set_title(f"12-lead: {col}", fontsize=16, fontweight="bold")
                ax.grid(True)
                plot_idx += 1

            # Hide any unused subplots in the grid
            for i in range(plot_idx, len(axes)):
                axes[i].set_visible(False)

            plt.tight_layout(rect=[0, 0, 1, 0.95])

            #
            # --- Comment: Save the combined plot to a file ---
            #
            output_filename = os.path.join(
                output_plot_dir, os.path.basename(file_path).replace(".csv", ".png")
            )
            plt.savefig(output_filename)
            plt.close(fig)

        except Exception as e:
            print(f"    - Error plotting {file_path}: {e}")

    print(
        f"    - Finished plotting for {subject_name}. Found in 'individual_plots' subdir."
    )


def main():
    """
    Main function to orchestrate the analysis for different base channels.
    """
    full_dataset_path = os.path.join(BASE_DATA_DIR, DATASET_NAME)

    if not os.path.isdir(full_dataset_path):
        print(f"Error: Main dataset directory not found at '{full_dataset_path}'")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    base_channels_to_process = ["0", "center_1", "center_2", "center_3", "center_4"]
    for base_ch in base_channels_to_process:
        run_analysis_for_base_ch(full_dataset_path, base_ch)

    print("\nAnalysis complete for all specified base channels.")


if __name__ == "__main__":
    main()
