import os
import sys
import argparse
import pandas as pd
import numpy as np
from scipy.interpolate import griddata
from glob import glob
from tqdm import tqdm
import warnings

# Suppress specific warnings from griddata
warnings.filterwarnings("ignore", category=UserWarning, message=".*RectBivariateSpline.*\)")

def get_sensor_coordinates():
    """
    Returns the 2D coordinates for the 16 sensors based on Pattern B.
    Returns:
        dict: A dictionary mapping channel names ('ch_1' to 'ch_16') to (x, y) tuples.
    """
    coords = {}
    for i in range(16):
        col = i // 4
        row = 3 - (i % 4)
        coords[f'ch_{i+1}'] = (col, row)
    return coords

def apply_rotation_correction(df, axis_degrees, sensor_coords, base_ch_name):
    """
    Applies rotation correction to the 16-channel data within the dataframe.
    """
    # --- 1. Prepare data ---
    theta_rad = np.deg2rad(axis_degrees)
    
    # Extract original 16 channel data and their coordinates
    ch_names = [f'ch_{i+1}' for i in range(16)]
    if not all(ch in df.columns for ch in ch_names):
        # This dataset seems to be a 15-channel differential one. We cannot reconstruct the original 16ch.
        # This function requires the raw 16-channel data before differentiation.
        print("Warning: Raw 16-channel data not found. Skipping rotation correction.", file=sys.stderr)
        return None

    original_16ch_data = df[ch_names]
    original_coords = np.array([sensor_coords[ch] for ch in ch_names])
    
    # --- 2. Calculate new coordinates by rotating the grid ---
    # We want to rotate the *measurement grid* by -theta to align the axis to 0.
    # This is equivalent to keeping the grid fixed and asking for values at points rotated by +theta.
    # However, the user's request is to correct the *data*, which means transforming
    # the data from the rotated frame to the canonical frame.
    # Let (x, y) be the canonical coordinates. The rotated sensor at (x_rot, y_rot) measures the potential.
    # We want to find the potential at (x, y).
    # x_rot = x*cos(theta) - y*sin(theta)
    # y_rot = x*sin(theta) + y*cos(theta)
    # The potential at (x,y) in the canonical frame is the potential measured at (x_rot, y_rot)
    # in the rotated frame.
    # So, for each channel's canonical position, we find its corresponding rotated coordinate
    # and interpolate the value from the original 16-channel data.
    
    corrected_16ch_data = pd.DataFrame(index=df.index)
    
    # --- 3. Create target coordinates for interpolation ---
    # For each channel's canonical position, find where it *came from* in the rotated grid.
    target_coords = []
    cos_theta = np.cos(-theta_rad)
    sin_theta = np.sin(-theta_rad)
    
    for ch in ch_names:
        x, y = sensor_coords[ch]
        x_prime = x * cos_theta - y * sin_theta
        y_prime = x * sin_theta + y * cos_theta
        target_coords.append([x_prime, y_prime])
    
    target_coords = np.array(target_coords)

    # --- 4. Interpolate all time-series data at once ---
    # griddata expects values as (n_points, n_values). Here, (16 channels, n_samples).
    values = original_16ch_data.to_numpy().T
    
    # Perform interpolation for all time steps
    interpolated_values = griddata(original_coords, values, target_coords, method='cubic')
    
    # Handle potential NaNs from interpolation (points outside the convex hull)
    # by falling back to the nearest neighbor.
    nan_mask = np.isnan(interpolated_values).any(axis=1)
    if nan_mask.any():
        print(f"Warning: {nan_mask.sum()} points were outside the interpolation hull. Using 'nearest' method for those.", file=sys.stderr)
        nearest_values = griddata(original_coords, values, target_coords[nan_mask], method='nearest')
        interpolated_values[nan_mask] = nearest_values

    # Transpose back to (n_samples, 16 channels)
    corrected_16ch_np = interpolated_values.T
    
    # Create a new dataframe for the corrected data
    df_corrected_16ch = pd.DataFrame(corrected_16ch_np, columns=ch_names, index=df.index)

    # --- 5. Create the final 15-channel differential dataset ---
    base_series = df_corrected_16ch[base_ch_name]
    other_ch_names = [ch for ch in ch_names if ch != base_ch_name]
    other_df = df_corrected_16ch[other_ch_names]
    
    diff_df = other_df.subtract(base_series, axis=0)
    
    # --- 6. Combine with other data (Time, 12-lead, etc.) ---
    non_16ch_cols = [col for col in df.columns if col not in ch_names]
    final_df = pd.concat([df[non_16ch_cols], diff_df], axis=1)
    
    return final_df

def main(args):
    # 1. Read the cardiac axis file
    if not os.path.exists(args.axis_file):
        print(f"Error: Cardiac axis file not found at {args.axis_file}", file=sys.stderr)
        sys.exit(1)
        
    axis_df = pd.read_csv(args.axis_file)
    # Use the 15-channel axis for correction
    axis_degrees = axis_df['cardiac_axis_15ch_degrees'].iloc[0]
    subject_name = axis_df['subject'].iloc[0]
    print(f"Found cardiac axis for {subject_name}: {axis_degrees:.2f} degrees")

    # 2. Get sensor coordinates
    sensor_coords = get_sensor_coordinates()
    
    # 3. Define input and output directories
    create_directory_if_not_exists(args.output_dir)
    print(f"Outputting corrected files to: {args.output_dir}")

    # 4. Find all dataset files to process
    files_to_process = sorted(glob(os.path.join(args.data_dir, "dataset_*.csv")))
    if not files_to_process:
        print(f"No dataset files found in {args.data_dir}", file=sys.stderr)
        sys.exit(1)

    # 5. Process each file
    base_ch_name = f"ch_{args.base_ch_num}"
    for file_path in tqdm(files_to_process, desc="Applying rotation correction"):
        try:
            df = pd.read_csv(file_path)
            
            # Apply the correction
            df_corrected = apply_rotation_correction(df, axis_degrees, sensor_coords, base_ch_name)
            
            if df_corrected is not None:
                # Save the new dataframe
                output_filename = os.path.basename(file_path)
                output_filepath = os.path.join(args.output_dir, output_filename)
                df_corrected.to_csv(output_filepath, index=False, float_format='%.6f')

        except Exception as e:
            print(f"Error processing file {file_path}: {e}", file=sys.stderr)
            continue
            
    print("\nProcessing complete.")

def create_directory_if_not_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply rotation correction to datasets based on a calculated cardiac axis.")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the directory containing the source dataset_*.csv files.")
    parser.add_argument("--axis_file", type=str, required=True, help="Path to the cardiac_axis.csv file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the directory where corrected files will be saved.")
    parser.add_argument("--base_ch_num", type=int, default=1, help="The base channel number that will be subtracted to create the final 15-channel data (default: 1).")
    
    args = parser.parse_args()
    main(args)
