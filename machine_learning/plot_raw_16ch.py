import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys

# Add the project root to the Python path to import settings if needed
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

# We will use the user-provided sampling rate, but keep RATE as a fallback.
try:
    from config.settings import RATE
except ImportError:
    RATE = 500 # Default fallback if settings.py is not found

def plot_raw_16ch_waveform(input_file, output_file, sampling_rate, num_channels=16):
    """
    Plots raw 16-channel waveform data from a CSV file onto a single graph.
    All channels are overlaid. The plot is optimized for thesis publication.

    Args:
        input_file (str): Path to the input CSV file.
        output_file (str): Path to save the output plot (e.g., .svg).
        sampling_rate (float): Sampling rate of the data in Hz.
        num_channels (int): Number of channels to plot.
    """
    if not os.path.exists(input_file):
        print(f"Error: Input file not found at '{input_file}'")
        sys.exit(1)

    try:
        df = pd.read_csv(input_file, header=None)
        waveform_data = df.iloc[:, :num_channels].to_numpy()
        time_axis = np.arange(waveform_data.shape[0]) / sampling_rate

        # --- Plotting Logic for Single Overlaid Figure (Horizontally Long) ---
        fig, ax = plt.subplots(figsize=(22, 9))

        colors = plt.cm.get_cmap('tab20')

        for i in range(num_channels):
            ax.plot(time_axis, waveform_data[:, i], lw=1.5, color=colors(i % 20), label=f"Channel {i+1}", alpha=0.7)

        # --- Aesthetics for Thesis Publication ---
        ax.set_xlabel("Time (s)", fontsize=22) # Increased font size
        ax.set_ylabel("Amplitude", fontsize=22) # Increased font size
        ax.tick_params(axis='both', which='major', labelsize=18) # Increased font size
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Set x-axis limit to 5 seconds
        ax.set_xlim(0, 5)
        
        # Place legend in the upper right corner inside the plot
        ax.legend(loc='upper right', fontsize=16) # Increased font size
        
        fig.suptitle("Raw 16-Channel Sensor Waveforms (Takahashi Test)", fontsize=28, y=0.98) # Increased font size
        
        plt.tight_layout()

        # Ensure output directory exists
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Plot saved to '{output_file}'")

    except Exception as e:
        print(f"An error occurred while plotting: {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot raw 16-channel waveform data from a CSV file for master's thesis."
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to the input CSV file containing raw waveform data."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to save the output plot (e.g., path/to/plot.svg)."
    )
    parser.add_argument(
        "--sampling_rate",
        type=float,
        default=RATE,
        help=f"Sampling rate of the data in Hz."
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=16,
        help="Number of channels to plot (default: 16)."
    )

    args = parser.parse_args()

    plot_raw_16ch_waveform(
        args.input_file,
        args.output_file,
        args.sampling_rate,
        args.channels
    )
