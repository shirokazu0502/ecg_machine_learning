import argparse
import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_comparison(original_file, corrected_file, channel, output_image):
    """
    Compares a channel from the original and corrected files by plotting them.
    """
    if not os.path.exists(original_file):
        print(f"Error: Original file not found at {original_file}")
        return
    if not os.path.exists(corrected_file):
        print(f"Error: Corrected file not found at {corrected_file}")
        return

    try:
        df_orig = pd.read_csv(original_file)
        df_corr = pd.read_csv(corrected_file)

        if channel not in df_orig.columns:
            print(f"Error: Channel '{channel}' not found in original file.")
            return
        if channel not in df_corr.columns:
            print(f"Error: Channel '{channel}' not found in corrected file.")
            return

        plt.figure(figsize=(15, 7))
        
        # Use 'Time' column if available, otherwise use index
        time_axis_orig = df_orig.get('Time', df_orig.index)
        time_axis_corr = df_corr.get('Time', df_corr.index)

        plt.plot(time_axis_orig, df_orig[channel], label=f'Original - {channel}', alpha=0.8)
        plt.plot(time_axis_corr, df_corr[channel], label=f'Corrected - {channel}', alpha=0.8, linestyle='--')
        
        plt.title(f'Comparison of Channel "{channel}" Before and After Rotation Correction')
        plt.xlabel("Time (s) or Index")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.grid(True)
        
        plt.savefig(output_image)
        print(f"Comparison plot saved to {output_image}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot comparison of original and corrected ECG data.")
    parser.add_argument("--original_file", type=str, required=True, help="Path to the original dataset CSV file.")
    parser.add_argument("--corrected_file", type=str, required=True, help="Path to the corrected dataset CSV file.")
    parser.add_argument("--channel", type=str, required=True, help="The channel name to plot (e.g., 'ch_2').")
    parser.add_argument("--output_image", type=str, required=True, help="Path to save the output plot image.")
    
    args = parser.parse_args()
    plot_comparison(args.original_file, args.corrected_file, args.channel, args.output_image)
