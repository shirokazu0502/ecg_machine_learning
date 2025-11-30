import os
import sys
import pandas as pd
import numpy as np
from glob import glob
from tqdm import tqdm

def get_channel_coords():
    """
    Returns the 2D coordinates for the 16 sensors, assuming ch_1 is top-left.
    This corresponds to the 'flipped' orientation.
    Grid: (row, column)
    """
    coords = {}
    for i in range(16):  # i is 0-indexed
        row = i // 4
        col = i % 4
        coords[f"ch_{i+1}"] = (row, col)
    return coords

def classify_orientation(vector):
    """
    Classifies the grid orientation based on the dipole vector.
    Assumes the anatomical heart axis vector is generally Down and Left.
    Grid coordinates: row increases downwards, col increases to the right.
    - A 'flipped' grid (ch_1 top-left) should have a Down-Right vector on the grid.
    - A 'normal' grid (ch_1 bottom-right) should have an Up-Left vector on the grid.
    """
    dy, dx = vector
    
    if abs(dx) < 0.5 and abs(dy) < 0.5:
        return "Uncertain (Zero Vector)"
        
    if dy > 0 and dx > 0:
        return "Down-Right (Likely Flipped)"
    elif dy < 0 and dx < 0:
        return "Up-Left (Likely Normal)"
    elif dy > 0 and dx < 0:
        return "Down-Left"
    elif dy < 0 and dx > 0:
        return "Up-Right"
    elif abs(dx) > abs(dy):
        return "Horizontal Dominant"
    else:
        return "Vertical Dominant"

def estimate_subject_orientation(subject_dir):
    """
    Estimates the sensor grid orientation for a single subject.
    """
    try:
        # --- 1. Create Average Heartbeat ---
        files_to_process = sorted(glob(os.path.join(subject_dir, "0", "dataset_*.csv")))
        if not files_to_process:
            return {"subject": os.path.basename(subject_dir), "error": "No dataset files found"}

        all_beats = [pd.read_csv(f) for f in files_to_process]
        avg_beat_df = pd.concat(all_beats).groupby(level=0).mean()
        
        phys_channels = [f"ch_{i}" for i in range(1, 17)]
        avg_beat_16ch = avg_beat_df[phys_channels]

        # --- 2. Find R-peak time via Global Field Power (GFP) ---
        gfp = avg_beat_16ch.std(axis=1)
        t_r_peak = gfp.idxmax()

        # --- 3. Get Potential Map at R-peak ---
        potential_map = avg_beat_16ch.loc[t_r_peak]

        # --- 4. Find Min/Max Channels ---
        ch_min = potential_map.idxmin()
        ch_max = potential_map.idxmax()

        # --- 5. Determine Dipole Vector ---
        channel_coords = get_channel_coords()
        coord_min = np.array(channel_coords[ch_min])
        coord_max = np.array(channel_coords[ch_max])
        vector = coord_max - coord_min  # (dy, dx)

        # --- 6. Classify Orientation ---
        orientation = classify_orientation(vector)

        return {
            "subject": os.path.basename(subject_dir),
            "ch_min": ch_min,
            "ch_max": ch_max,
            "dipole_vector_dy_dx": str(vector),
            "estimated_orientation": orientation,
            "error": None
        }

    except Exception as e:
        return {"subject": os.path.basename(subject_dir), "error": str(e)}


def main():
    project_root = ".."
    input_root = os.path.join(project_root, "data", "processed", "for_best_resample_base_max_amp")
    output_file = os.path.join(project_root, "data", "processed", "estimated_orientations_max_amp.csv")
    
    try:
        all_items = os.listdir(input_root)
        subject_list = sorted([item for item in all_items if os.path.isdir(os.path.join(input_root, item))])
    except FileNotFoundError:
        print(f"Error: Input directory not found at {input_root}")
        sys.exit(1)

    if not subject_list:
        print("No subject directories found. Exiting.")
        sys.exit(0)

    print(f"Found {len(subject_list)} subjects. Estimating orientation for each...")

    results = []
    for subject_name in tqdm(subject_list, desc="Processing Subjects"):
        subject_dir = os.path.join(input_root, subject_name)
        result = estimate_subject_orientation(subject_dir)
        results.append(result)

    # Save results to CSV
    results_df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    results_df.to_csv(output_file, index=False)

    print(f"\nProcessing complete. Results saved to: {output_file}")
    print("\n--- Results Summary ---")
    print(results_df)


if __name__ == "__main__":
    main()
