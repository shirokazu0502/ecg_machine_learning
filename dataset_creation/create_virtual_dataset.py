import os
import sys
import argparse
import pandas as pd
import numpy as np
from glob import glob
from tqdm import tqdm

# Add base directory to sys.path
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.append(project_root)


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
    physical_coords = get_sensor_coordinates(orientation)
    coord_to_ch = {v: k for k, v in physical_coords.items()}

    virtual_df = pd.DataFrame(index=df_16ch.index)

    v_idx = 0
    # Iterate through the 3x3 possible virtual electrode positions
    for y_base in range(2, -1, -1):
        for x_base in range(3):
            v_idx += 1
            v_name = f"v_ch_{v_idx}"

            ch_tl = coord_to_ch.get((x_base, y_base + 1))
            ch_tr = coord_to_ch.get((x_base + 1, y_base + 1))
            ch_bl = coord_to_ch.get((x_base, y_base))
            ch_br = coord_to_ch.get((x_base + 1, y_base))

            channels_to_average = [
                ch for ch in [ch_tl, ch_tr, ch_bl, ch_br] if ch is not None
            ]
            virtual_df[v_name] = df_16ch[channels_to_average].mean(axis=1)

    return virtual_df


def process_directory(input_dir, output_dir, orientation="normal"):
    """
    Converts all 16-ch dataset files in a directory to 9-ch virtual electrode files.
    """
    create_directory_if_not_exists(output_dir)
    print(f"Processing directory: {input_dir}")
    print(f"Outputting virtual electrode files to: {output_dir}")

    files_to_process = sorted(glob(os.path.join(input_dir, "dataset_*.csv")))
    if not files_to_process:
        print(f"No dataset files found in {input_dir}", file=sys.stderr)
        return

    physical_ch_names = [f"ch_{i+1}" for i in range(16)]

    for file_path in tqdm(
        files_to_process, desc=f"Converting {os.path.basename(input_dir)}"
    ):
        try:
            df = pd.read_csv(file_path)

            # Ensure 16-ch data exists
            if not all(ch in df.columns for ch in physical_ch_names):
                print(
                    f"Warning: 16-ch data not found in {file_path}. Skipping.",
                    file=sys.stderr,
                )
                continue

            # Separate 16ch data from other data (Time, 12-lead, etc.)
            df_16ch = df[physical_ch_names]
            df_non_16ch = df.drop(columns=physical_ch_names)

            # Separate Time and 12-lead data
            time_cols = [col for col in df_non_16ch.columns if "Time" in col]
            df_time = (
                df_non_16ch[time_cols] if time_cols else pd.DataFrame(index=df.index)
            )
            df_12lead = df_non_16ch.drop(columns=time_cols, errors="ignore")

            # Create 9-ch virtual electrode data
            df_virtual_9ch = create_virtual_electrodes(df_16ch, orientation)

            # Combine data in the desired order: Time, Virtual Electrodes, 12-Lead ECG
            df_final = pd.concat([df_time, df_virtual_9ch, df_12lead], axis=1)

            # Save the new dataframe
            output_filename = os.path.basename(file_path)
            output_filepath = os.path.join(output_dir, output_filename)
            df_final.to_csv(output_filepath, index=False, float_format="%.6f")

        except Exception as e:
            print(f"Error processing file {file_path}: {e}", file=sys.stderr)
            continue


def create_directory_if_not_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)


def main(args):
    # Process the main dataset directory
    process_directory(args.input_dir, args.output_dir, args.orientation)

    # Process the moving_ave_datasets subdirectory if it exists
    moving_ave_input_dir = os.path.join(args.input_dir, "moving_ave_datasets")
    if os.path.isdir(moving_ave_input_dir):
        moving_ave_output_dir = os.path.join(args.output_dir, "moving_ave_datasets")
        process_directory(moving_ave_input_dir, moving_ave_output_dir, args.orientation)

    print("\nProcessing complete.")


if __name__ == "__main__":
    # Arguments are set directly for simplicity
    class Args:
        pass

    args = Args()

    # ##################################################################
    # #############  CONFIGURATION FOR THE SCRIPT RUN  ###############
    # ##################################################################
    subject_name_full = "nishio_0513_0.8s"
    args.orientation = "normal"  # <--- CHANGE THIS: "normal" or "flipped"
    # ##################################################################

    # Input directory with original 16-channel data
    args.input_dir = os.path.join(
        project_root, "data", "processed", "for_best_resample", subject_name_full, "0"
    )
    # Output directory for the new 9-channel virtual electrode dataset
    output_subject_dir = f"{subject_name_full}_{args.orientation}"
    args.output_dir = os.path.join(
        project_root,
        "data",
        "processed",
        "virtual_electrode_dataset",
        output_subject_dir,
        "0",
    )

    main(args)
