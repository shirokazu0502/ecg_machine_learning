import os
import sys
import argparse
import pandas as pd
import numpy as np
from scipy.interpolate import griddata
from glob import glob
from tqdm import tqdm
import warnings
from scipy.spatial import distance

# Suppress specific warnings from griddata
warnings.filterwarnings(
    "ignore", category=UserWarning, message=".*RectBivariateSpline.*\)"
)


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
        coords[f"ch_{i+1}"] = (col, row)
    return coords


def apply_rotation_correction(df, axis_degrees, sensor_coords, base_ch_name):
    """
    Applies rotation correction to the 16-channel data within the dataframe
    using an extended grid to improve interpolation stability.
    """
    # --- 1. Prepare data ---
    theta_rad = np.deg2rad(axis_degrees)
    ch_names = [f"ch_{i+1}" for i in range(16)]
    if not all(ch in df.columns for ch in ch_names):
        print(
            "Warning: Raw 16-channel data not found. Skipping rotation correction.",
            file=sys.stderr,
        )
        return None

    original_16ch_data = df[ch_names]
    original_coords = np.array([sensor_coords[ch] for ch in ch_names])

    # --- 2. Create a 8x8 extended grid ---
    # Define the coordinates of the new 8x8 grid (from -2 to 5)
    x_ext = np.arange(-4, 8)
    y_ext = np.arange(-4, 8)
    xx_ext, yy_ext = np.meshgrid(x_ext, y_ext)
    extended_coords = np.vstack([xx_ext.ravel(), yy_ext.ravel()]).T

    # Find the nearest original sensor for each point in the extended grid
    dist_matrix = distance.cdist(extended_coords, original_coords)
    nearest_indices = np.argmin(dist_matrix, axis=1)

    # Create the extended dataset by copying data from the nearest sensor
    # original_16ch_data is (n_samples, 16), we need (n_samples, 64)
    extended_values_np = original_16ch_data.to_numpy()[:, nearest_indices]

    # --- 3. Calculate target coordinates for interpolation ---
    target_coords = []
    cos_theta = np.cos(-theta_rad)
    sin_theta = np.sin(-theta_rad)
    for ch in ch_names:
        x, y = sensor_coords[ch]
        x_prime = x * cos_theta - y * sin_theta
        y_prime = x * sin_theta + y * cos_theta
        target_coords.append([x_prime, y_prime])
    target_coords = np.array(target_coords)

    # --- 4. Interpolate using the extended grid ---
    # griddata expects values as (n_points, n_values). Here, (36 points, n_samples).
    values = extended_values_np.T

    interpolated_values = griddata(
        extended_coords, values, target_coords, method="cubic"
    )

    # Fallback for any remaining NaNs (should be rare with the extended grid)
    nan_mask = np.isnan(interpolated_values).any(axis=1)
    if nan_mask.any():
        print(
            f"Warning: {nan_mask.sum()} points still outside hull. Using 'nearest'.",
            file=sys.stderr,
        )
        nearest_values = griddata(
            extended_coords, values, target_coords[nan_mask], method="nearest"
        )
        interpolated_values[nan_mask] = nearest_values

    corrected_16ch_np = interpolated_values.T
    df_corrected_16ch = pd.DataFrame(
        corrected_16ch_np, columns=ch_names, index=df.index
    )

    # --- 5. Create the final 15-channel differential dataset ---
    base_series = df_corrected_16ch[base_ch_name]
    other_ch_names = [ch for ch in ch_names if ch != base_ch_name]
    other_df = df_corrected_16ch[other_ch_names]
    diff_df = other_df.subtract(base_series, axis=0)

    # --- 6. Combine with other data (Time, 12-lead, etc.) ---
    non_16ch_cols = [col for col in df.columns if col not in ch_names]
    time_col = [col for col in non_16ch_cols if "Time" in col]
    medical_ecg_cols = [col for col in non_16ch_cols if "Time" not in col]
    final_df = pd.concat([df[time_col], diff_df, df[medical_ecg_cols]], axis=1)

    return final_df


def process_directory(data_dir, output_dir, axis_degrees, sensor_coords, base_ch_name):
    """
    Applies rotation correction to all dataset files in a given directory.
    """
    create_directory_if_not_exists(output_dir)
    print(f"Processing directory: {data_dir}")
    print(f"Outputting corrected files to: {output_dir}")

    files_to_process = sorted(glob(os.path.join(data_dir, "dataset_*.csv")))
    if not files_to_process:
        print(f"No dataset files found in {data_dir}", file=sys.stderr)
        return

    for file_path in tqdm(
        files_to_process, desc=f"Correcting {os.path.basename(data_dir)}"
    ):
        try:
            df = pd.read_csv(file_path)

            # Apply the correction
            df_corrected = apply_rotation_correction(
                df, axis_degrees, sensor_coords, base_ch_name
            )

            if df_corrected is not None:
                # Save the new dataframe
                output_filename = os.path.basename(file_path)
                output_filepath = os.path.join(output_dir, output_filename)
                df_corrected.to_csv(output_filepath, index=False, float_format="%.6f")

        except Exception as e:
            print(f"Error processing file {file_path}: {e}", file=sys.stderr)
            continue


def main(args):
    # 1. Read the cardiac axis file
    if not os.path.exists(args.axis_file):
        print(
            f"Error: Cardiac axis file not found at {args.axis_file}", file=sys.stderr
        )
        sys.exit(1)

    axis_df = pd.read_csv(args.axis_file)
    axis_degrees = axis_df["cardiac_axis_15ch_degrees"].iloc[0]
    subject_name = axis_df["subject"].iloc[0]
    print(f"Found cardiac axis for {subject_name}: {axis_degrees:.2f} degrees")

    # 2. Get sensor coordinates
    sensor_coords = get_sensor_coordinates()
    base_ch_name = f"ch_{args.base_ch_num}"

    # 3. Process the main dataset directory
    process_directory(
        args.data_dir, args.output_dir, axis_degrees, sensor_coords, base_ch_name
    )

    # 4. Process the moving_ave_datasets subdirectory if it exists
    moving_ave_dir = os.path.join(args.data_dir, "moving_ave_datasets")
    if os.path.isdir(moving_ave_dir):
        moving_ave_output_dir = os.path.join(args.output_dir, "moving_ave_datasets")
        process_directory(
            moving_ave_dir,
            moving_ave_output_dir,
            axis_degrees,
            sensor_coords,
            base_ch_name,
        )

    print("\nProcessing complete.")


def create_directory_if_not_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Apply rotation correction to datasets based on a calculated cardiac axis."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to the directory containing the source dataset_*.csv files.",
    )
    parser.add_argument(
        "--axis_file",
        type=str,
        required=True,
        help="Path to the cardiac_axis.csv file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to the directory where corrected files will be saved.",
    )
    parser.add_argument(
        "--base_ch_num",
        type=int,
        default=1,
        help="The base channel number that will be subtracted to create the final 15-channel data (default: 1).",
    )

    args = parser.parse_args()
    main(args)
