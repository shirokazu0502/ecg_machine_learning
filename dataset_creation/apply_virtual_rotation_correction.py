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


def get_sensor_coordinates(orientation="normal"):
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
            raise ValueError("Invalid orientation specified. Use 'normal' or 'flipped'.")
        
        # coords key is 1-indexed
        coords[f"ch_{i+1}"] = (col, row)
    return coords


def get_virtual_sensor_coordinates():
    """


    Returns the 2D coordinates for the 9 virtual sensors based on their


    absolute positions on the physical 4x4 grid (0-3).


    The center of the grid is (1.5, 1.5).


    """

    coords = {}

    v_idx = 0

    # y_center moves from 2.5 down to 0.5

    for y_center in np.arange(2.5, 0.0, -1.0):

        # x_center moves from 0.5 up to 2.5

        for x_center in np.arange(0.5, 3.0, 1.0):

            v_idx += 1

            coords[f"v_ch_{v_idx}"] = (x_center, y_center)

    return coords


def rotate_points(points, angle_deg, center):
    """Rotates a list of points around a given center."""
    angle_rad = np.deg2rad(angle_deg)
    cos_theta = np.cos(angle_rad)
    sin_theta = np.sin(angle_rad)
    center_x, center_y = center
    rotated_points = []
    for x, y in points:
        x_translated = x - center_x
        y_translated = y - center_y
        x_rotated = x_translated * cos_theta - y_translated * sin_theta
        y_rotated = x_translated * sin_theta + y_translated * cos_theta
        x_final = x_rotated + center_x
        y_final = y_rotated + center_y
        rotated_points.append([x_final, y_final])
    return np.array(rotated_points)


def apply_rotation_correction(df, axis_degrees, physical_sensor_coords):
    """
    Applies rotation correction using a two-stage interpolation process based on
    a precise physical model.
    """
    # --- 1. Prepare Original 16-channel data ---
    physical_ch_names = list(physical_sensor_coords.keys())
    if not all(ch in df.columns for ch in physical_ch_names):
        print("Warning: Raw 16-channel data not found. Skipping.", file=sys.stderr)
        return None

    original_16ch_data = df[physical_ch_names]
    original_16ch_coords = np.array(list(physical_sensor_coords.values()))
    original_16ch_values = original_16ch_data.to_numpy()

    # --- 2. Stage 1: Create a High-Resolution Potential Map ---
    # Define a 17x17 intermediate fine grid (from 0 to 3)
    grid_x, grid_y = np.mgrid[0:3:17j, 0:3:17j]
    fine_grid_coords = np.vstack([grid_x.ravel(), grid_y.ravel()]).T

    # Interpolate the 16-ch data onto the fine grid for each time step.
    fine_grid_values = griddata(
        original_16ch_coords,
        original_16ch_values.T,
        fine_grid_coords,
        method="cubic",
    )

    # Fallback for NaNs in the fine grid itself
    nan_mask_fine = np.isnan(fine_grid_values).any(axis=1)
    if nan_mask_fine.any():
        nearest_fine_values = griddata(
            original_16ch_coords,
            original_16ch_values.T,
            fine_grid_coords[nan_mask_fine],
            method="nearest",
        )
        fine_grid_values[nan_mask_fine] = nearest_fine_values

    # --- 3. Calculate Rotated Target Coordinates ---
    # Get the absolute pre-rotation coordinates of virtual electrodes
    virtual_sensor_coords = get_virtual_sensor_coordinates()
    virtual_ch_names = list(virtual_sensor_coords.keys())
    original_virtual_coords = np.array(list(virtual_sensor_coords.values()))

    # Rotate these coordinates around the center of the physical grid (1.5, 1.5)
    target_coords = rotate_points(
        points=original_virtual_coords,
        angle_deg=-axis_degrees,  # Apply counter-rotation
        center=(1.5, 1.5),
    )

    # --- 4. Stage 2: Interpolate Final Values from the High-Res Map ---
    interpolated_values = griddata(
        fine_grid_coords, fine_grid_values, target_coords, method="linear"
    )

    # Fallback for the final interpolation
    nan_mask_final = np.isnan(interpolated_values).any(axis=1)
    if nan_mask_final.any():
        print(
            f"Warning: {nan_mask_final.sum()} final points outside hull. Using 'nearest'.",
            file=sys.stderr,
        )
        nearest_final_values = griddata(
            fine_grid_coords,
            fine_grid_values,
            target_coords[nan_mask_final],
            method="nearest",
        )
        interpolated_values[nan_mask_final] = nearest_final_values

    # --- 5. Create Final DataFrame ---
    # Transpose back to (n_samples, n_channels)
    corrected_9ch_np = interpolated_values.T
    df_corrected_9ch = pd.DataFrame(
        corrected_9ch_np, columns=virtual_ch_names, index=df.index
    )

    # Combine with other data in the desired order
    non_16ch_cols = [col for col in df.columns if col not in physical_ch_names]
    df_non_16ch = df[non_16ch_cols]

    time_cols = [col for col in df_non_16ch.columns if "Time" in col]
    df_time = df_non_16ch[time_cols] if time_cols else pd.DataFrame(index=df.index)
    df_12lead = df_non_16ch.drop(columns=time_cols, errors='ignore')

    final_df = pd.concat([df_time, df_corrected_9ch, df_12lead], axis=1)

    return final_df


def process_directory(data_dir, output_dir, axis_degrees, orientation="normal"):
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

    physical_sensor_coords = get_sensor_coordinates(orientation)

    for file_path in tqdm(
        files_to_process, desc=f"Correcting {os.path.basename(data_dir)}"
    ):
        try:
            df = pd.read_csv(file_path)

            # Apply the correction
            df_corrected = apply_rotation_correction(
                df, axis_degrees, physical_sensor_coords
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
    axis_degrees = axis_df["cardiac_axis_virtual_degrees"].iloc[0]
    subject_name = axis_df["subject"].iloc[0]
    print(f"Found cardiac axis for {subject_name}: {axis_degrees:.2f} degrees")

    # 2. Process the main dataset directory
    process_directory(
        args.data_dir, args.output_dir, axis_degrees, args.orientation
    )

    # 3. Process the moving_ave_datasets subdirectory if it exists
    moving_ave_dir = os.path.join(args.data_dir, "moving_ave_datasets")
    if os.path.isdir(moving_ave_dir):
        moving_ave_output_dir = os.path.join(args.output_dir, "moving_ave_datasets")
        process_directory(
            moving_ave_dir,
            moving_ave_output_dir,
            axis_degrees,
            args.orientation,
        )

    print("\nProcessing complete.")


def create_directory_if_not_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)


if __name__ == "__main__":
    # Arguments are now set directly within the script
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

    # Define subject dir with orientation
    input_subject_dir = f"{subject_name_full}_{args.orientation}"

    args.data_dir = os.path.join(
        project_root, "data", "processed", "for_best_resample", subject_name_full, "0"
    )
    args.axis_file = os.path.join(project_root, "data", "processed", "virtual_electrode_dataset", input_subject_dir, "virtual_axis.csv")
    args.output_dir = os.path.join(
        project_root,
        "data",
        "processed",
        "rotated_datasets",
        f"{subject_name_full}_virtual_rotated_{args.orientation}",
    )

    main(args)
