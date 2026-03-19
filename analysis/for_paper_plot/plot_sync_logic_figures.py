import argparse
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import neurokit2 as nk

# Add project root to path to allow importing config
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

# --- Utility Functions ---

def filter_ecg(data, sampling_rate, highpass_hz=0.5, notch_hz=50):
    """Applies high-pass and notch filters to the signal using scipy."""
    # High-pass filter (Butterworth)
    # Using nk.signal_filter for HPF as it's simple and common
    hp_filtered = nk.signal_filter(data, sampling_rate=sampling_rate, lowcut=highpass_hz, method='butterworth', order=5)
    
    # Notch filter using scipy.signal.iirnotch and filtfilt
    nyquist = sampling_rate / 2.0
    w0 = notch_hz / nyquist
    Q = 30 # Quality factor, a common value for ECG
    b_notch, a_notch = signal.iirnotch(w0=w0, Q=Q)
    filtered = signal.filtfilt(b_notch, a_notch, hp_filtered)
    
    return filtered

# --- Figure 1: Pre-processing Steps ---

def plot_figure1_preprocessing(sensor_file_path, output_path, sampling_rate_orig, sampling_rate_target, channel_to_plot=0):
    """
    Creates a 3-panel plot illustrating the pre-processing steps.
    """
    print("Generating Figure 1: Pre-processing steps...")
    try:
        # 1. Load Data
        df_sensor = pd.read_csv(sensor_file_path, header=None)
        raw_signal = df_sensor.iloc[:, channel_to_plot].to_numpy()
        
        # Limit to first 10 seconds for clarity
        num_samples_raw = int(10 * sampling_rate_orig)
        raw_signal = raw_signal[:num_samples_raw]
        time_axis_raw = np.arange(len(raw_signal)) / sampling_rate_orig

        # 2. Apply Filters
        filtered_signal = filter_ecg(raw_signal, sampling_rate=sampling_rate_orig)

        # 3. Resample
        num_samples_resampled = int(len(filtered_signal) * (sampling_rate_target / sampling_rate_orig))
        resampled_signal = signal.resample(filtered_signal, num_samples_resampled)
        time_axis_resampled = np.arange(len(resampled_signal)) / sampling_rate_target
        
        # 4. Create Plot
        fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
        
        # Panel 1: Raw Signal
        axes[0].plot(time_axis_raw, raw_signal, label="Raw Sensor Signal")
        axes[0].set_title("1. Raw Waveform", fontsize=16)
        axes[0].set_ylabel("Amplitude", fontsize=14)
        axes[0].grid(True, linestyle='--', alpha=0.6)
        
        # Panel 2: Filtered Signal
        axes[1].plot(time_axis_raw, filtered_signal, label="Filtered Signal", color='g')
        axes[1].set_title("2. After High-pass (0.5Hz) and Notch (50Hz) Filtering", fontsize=16)
        axes[1].set_ylabel("Amplitude", fontsize=14)
        axes[1].grid(True, linestyle='--', alpha=0.6)

        # Panel 3: Resampled Signal
        axes[2].plot(time_axis_resampled, resampled_signal, label="Resampled Signal", color='r')
        axes[2].set_title(f"3. After Resampling to {sampling_rate_target}Hz", fontsize=16)
        axes[2].set_xlabel("Time (s)", fontsize=14)
        axes[2].set_ylabel("Amplitude", fontsize=14)
        axes[2].grid(True, linestyle='--', alpha=0.6)
        
        for ax in axes:
            ax.tick_params(axis='both', which='major', labelsize=12)
        
        fig.suptitle("Figure 1: Pre-processing of Sensor Waveform", fontsize=20, y=1.0)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Figure 1 saved to '{output_path}'")

    except Exception as e:
        print(f"Error creating Figure 1: {e}")

# --- Figure 2: Synchronization and Segmentation ---

def plot_figure2_synchronization(sensor_file_path, ecg_file_path, output_path, sampling_rate_orig, sampling_rate_target, sensor_ch=0, ecg_ch=1):
    """
    Creates a 2-part plot illustrating synchronization and segmentation.
    """
    print("Generating Figure 2: Synchronization and Segmentation...")
    try:
        # 1. Load and Pre-process Sensor Data
        df_sensor = pd.read_csv(sensor_file_path, header=None)
        raw_sensor_signal = df_sensor.iloc[:, sensor_ch].to_numpy()
        filtered_sensor = filter_ecg(raw_sensor_signal, sampling_rate=sampling_rate_orig)
        num_samples_resampled = int(len(filtered_sensor) * (sampling_rate_target / sampling_rate_orig))
        sensor_500hz = signal.resample(filtered_sensor, num_samples_resampled)

        # 2. Load and Pre-process 12-lead ECG Data
        df_ecg = pd.read_csv(ecg_file_path, header=None)
        # Assuming Lead II is the second column (index 1)
        raw_ecg_signal = df_ecg.iloc[:, ecg_ch].to_numpy()
        ecg_500hz = filter_ecg(raw_ecg_signal, sampling_rate=sampling_rate_target)

        # Limit to a reasonable duration for visualization (e.g., 10 seconds)
        duration = 10
        sensor_500hz = sensor_500hz[:duration * sampling_rate_target]
        ecg_500hz = ecg_500hz[:duration * sampling_rate_target]

        # 3. Find R-peaks for both signals
        _, rpeaks_info_sensor = nk.ecg_peaks(sensor_500hz, sampling_rate=sampling_rate_target)
        _, rpeaks_info_ecg = nk.ecg_peaks(ecg_500hz, sampling_rate=sampling_rate_target)

        # --- DEEPER DEBUGGING ---
        print("--- Start Debug Info for ECG Peaks ---")
        print(f"rpeaks_info_ecg dictionary: {rpeaks_info_ecg}")
        if 'ECG_R_Peaks' in rpeaks_info_ecg:
            ecg_peaks_raw = rpeaks_info_ecg['ECG_R_Peaks']
            print(f"Raw ecg_peaks object type: {type(ecg_peaks_raw)}")
            if isinstance(ecg_peaks_raw, np.ndarray):
                print(f"ecg_peaks dtype: {ecg_peaks_raw.dtype}")
                print(f"ecg_peaks content: {ecg_peaks_raw}")
        print("--- End Debug Info ---")
        # --- END DEEPER DEBUGGING ---
        
        # Extract R-peak indices, ensuring they are valid NumPy arrays even if no peaks are found
        sensor_peaks = rpeaks_info_sensor.get('ECG_R_Peaks', np.array([]))
        ecg_peaks = rpeaks_info_ecg.get('ECG_R_Peaks', np.array([]))

        if sensor_peaks.size == 0 or ecg_peaks.size == 0:
            print("Error: Could not find R-peaks in one or both signals. Cannot generate Figure 2.")
            return

        # 4. Create Plot
        fig = plt.figure(figsize=(20, 14))
        gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.2])

        # --- Part A: Synchronization ---
        ax1 = fig.add_subplot(gs[0])
        time_axis = np.arange(len(ecg_500hz)) / sampling_rate_target
        
        # Calculate offset based on the first peak
        offset = ecg_peaks[0] - sensor_peaks[0]
        time_axis_sensor_shifted = (np.arange(len(sensor_500hz)) + offset) / sampling_rate_target
        
        ax1.plot(time_axis, ecg_500hz, label="Standard ECG (Lead II)", alpha=0.8)
        ax1.plot(time_axis_sensor_shifted, sensor_500hz, label="Proposed Sensor (Channel 1, Aligned)", alpha=0.8)
        
        # Mark aligned peaks
        ax1.plot(time_axis[ecg_peaks], ecg_500hz[ecg_peaks], 'x', label="ECG R-Peaks", markersize=10, mew=2)
        ax1.plot(time_axis_sensor_shifted[sensor_peaks], sensor_500hz[sensor_peaks], 'o', label="Sensor R-Peaks", markersize=6, alpha=0.7)

        ax1.set_title("A: Synchronization of Signals using R-Peak Alignment", fontsize=16)
        ax1.set_xlabel("Time (s)", fontsize=14)
        ax1.set_ylabel("Amplitude", fontsize=14)
        ax1.legend(loc='upper right', fontsize=12)
        ax1.grid(True, linestyle='--', alpha=0.6)

        # --- Part B: Segmentation ---
        ax2 = fig.add_subplot(gs[1])
        
        # Pick a representative R-peak for segmentation visualization.
        # Add robustness for cases where there are fewer than 3 peaks or NaN values.
        r_peak_idx = -1
        if len(ecg_peaks) >= 3 and not np.isnan(ecg_peaks[2]):
            r_peak_idx = int(ecg_peaks[2])
        else:
            # If the 3rd peak doesn't exist or is NaN, find the first valid peak
            valid_peaks = ecg_peaks[~np.isnan(ecg_peaks)]
            if valid_peaks.size > 0:
                r_peak_idx = int(valid_peaks[0])
                print(f"Warning: Using first valid R-peak ({r_peak_idx}) for segmentation viz.")
            else:
                print("Error: No valid R-peaks found for segmentation. Cannot generate Figure 2.")
                return
        
        start_idx = r_peak_idx - 150
        end_idx = r_peak_idx + 250 # Total 400 points
        
        # Ensure plot window is within bounds
        plot_start = max(0, start_idx - 200)
        plot_end = min(len(ecg_500hz), end_idx + 200)

        ax2.plot(time_axis[plot_start:plot_end], ecg_500hz[plot_start:plot_end], label="Synchronized ECG")
        
        # Draw the segmentation window
        ylim = ax2.get_ylim()
        rect = plt.Rectangle((time_axis[start_idx], ylim[0]), 
                             width=(time_axis[end_idx] - time_axis[start_idx]), 
                             height=ylim[1] - ylim[0],
                             facecolor='red', alpha=0.2, label="400-Point Segment Window")
        ax2.add_patch(rect)
        
        # Mark the R-peak
        ax2.axvline(time_axis[r_peak_idx], color='k', linestyle='--', label=f"R-Peak (at point 150 of segment)")

        ax2.set_title("B: Segmentation of a 400-Point (0.8s) Window around an R-Peak", fontsize=16)
        ax2.set_xlabel("Time (s)", fontsize=14)
        ax2.set_ylabel("Amplitude", fontsize=14)
        ax2.legend(loc='upper right', fontsize=12)
        ax2.grid(True, linestyle='--', alpha=0.6)
        
        for ax in [ax1, ax2]:
             ax.tick_params(axis='both', which='major', labelsize=12)

        fig.suptitle("Figure 2: Synchronization and Segmentation Logic", fontsize=20, y=1.0)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Figure 2 saved to '{output_path}'")

    except Exception as e:
        print(f"Error creating Figure 2: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate figures to explain ECG signal synchronization logic.")
    parser.add_argument("--sensor_csv", type=str, required=True, help="Path to the raw 16-channel sensor data CSV.")
    parser.add_argument("--ecg_csv", type=str, required=True, help="Path to the raw 12-lead ECG data CSV.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the output figures.")
    parser.add_argument("--sensor_rate", type=float, default=122.06, help="Original sampling rate of the sensor data.")
    parser.add_argument("--target_rate", type=int, default=500, help="Target sampling rate for all signals.")
    
    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate Figure 1
    fig1_path = os.path.join(args.output_dir, "figure1_preprocessing.png")
    plot_figure1_preprocessing(args.sensor_csv, fig1_path, args.sensor_rate, args.target_rate)

    # Generate Figure 2
    fig2_path = os.path.join(args.output_dir, "figure2_synchronization.png")
    plot_figure2_synchronization(args.sensor_csv, args.ecg_csv, fig2_path, args.sensor_rate, args.target_rate)
