import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

# --- Configuration ---
BASE_DATA_PATH = "/mnt/ecg_project/data/processed/for_best_resample/"
SUB_PATH = "15ch_diff_from_ch_16/moving_ave_datasets"  # Corrected path
OUTPUT_DIR = "refactored/output/new_patient_plots"

# Base identifiers for patients. The script will search for directories starting with these names.
PATIENT_IDENTIFIERS = [
    "patient1",
    "patient2",
    "patient3",
    "patient4",
    "patient5",
    "patient6",
    "patient7",
    "patient8",
    "patient9",
    "patient10",
]

# Channel names for the 8 medical ECG leads
MEDICAL_ECG_CHANNELS = ["A1", "A2", "V1", "V2", "V3", "V4", "V5", "V6"]


def plot_waveforms_for_patient(patient_dir_path):
    """
    Finds all dataset CSVs for a given full patient directory path, loads them, and plots the waveforms.
    """
    patient_name = os.path.basename(patient_dir_path)
    print(f"--- Processing patient directory: {patient_name} ---")

    # Define paths
    patient_data_path = os.path.join(patient_dir_path, SUB_PATH)
    patient_output_path = os.path.join(OUTPUT_DIR, patient_name)

    # Create output directory for the patient
    os.makedirs(patient_output_path, exist_ok=True)

    # Find all dataset files
    dataset_files = sorted(glob.glob(os.path.join(patient_data_path, "dataset_*.csv")))

    if not dataset_files:
        print(f"  No dataset files found for {patient_name} at {patient_data_path}")
        return

    print(f"  Found {len(dataset_files)} dataset files. Generating plots...")

    for file_path in dataset_files:
        try:
            # Load data
            df = pd.read_csv(file_path)

            # Extract channel data
            # Input channels are assumed to be the 15 channels after 'Time' and 'hr'
            input_channels = df.iloc[:, 1:16]  # Columns 2 to 16 (15 channels)
            input_channel_names = input_channels.columns.tolist()

            # Medical ECG channels
            medical_channels = df[MEDICAL_ECG_CHANNELS]

            # --- Plotting ---
            num_input_channels = len(input_channel_names)
            num_medical_channels = len(MEDICAL_ECG_CHANNELS)
            total_channels = num_input_channels + num_medical_channels

            fig, axes = plt.subplots(5, 5, figsize=(25, 20))
            fig.suptitle(
                f"Waveforms for {patient_name} - {os.path.basename(file_path)}",
                fontsize=24,
            )

            axes_flat = axes.flatten()

            # Plot 15 input channels
            for i in range(num_input_channels):
                ax = axes_flat[i]
                ax.plot(input_channels.iloc[:, i])
                ax.set_title(f"Input: {input_channel_names[i]}")
                ax.grid(True)

            # Plot 8 medical ECG channels
            for i in range(num_medical_channels):
                ax = axes_flat[num_input_channels + i]
                ax.plot(medical_channels.iloc[:, i], color="green")
                ax.set_title(f"Medical: {MEDICAL_ECG_CHANNELS[i]}")
                ax.grid(True)

            # Hide unused subplots
            for i in range(total_channels, len(axes_flat)):
                axes_flat[i].set_visible(False)

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])

            # Save figure
            output_filename = os.path.basename(file_path).replace(".csv", ".png")
            save_path = os.path.join(patient_output_path, output_filename)
            plt.savefig(save_path)
            plt.close(fig)

        except Exception as e:
            print(f"  Error processing {file_path}: {e}")

    print(
        f"  Finished plotting for {patient_name}. Plots saved in {patient_output_path}"
    )


if __name__ == "__main__":
    print("Starting waveform plotting for new patients...")

    # Dynamically find patient directories
    all_found_patients = []
    for identifier in PATIENT_IDENTIFIERS:
        # Find directories that start with the identifier
        found_dirs = glob.glob(os.path.join(BASE_DATA_PATH, identifier + "*"))
        if found_dirs:
            all_found_patients.extend(found_dirs)
        else:
            print(f"  > No directory found starting with '{identifier}'")

    if not all_found_patients:
        print("No patient directories found. Exiting.")
    else:
        print(f"Found {len(all_found_patients)} patient directories to process.")
        for patient_dir in all_found_patients:
            plot_waveforms_for_patient(patient_dir)

    print("--- All patients processed. ---")
