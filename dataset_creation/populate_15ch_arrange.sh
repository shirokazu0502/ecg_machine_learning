#!/bin/bash

# This script populates the '15ch_arrange_direction' dataset by copying the correct
# differential data from the 'for_best_resample' directory for each subject.

# Define base paths
# WARNING: These paths are absolute and based on the project structure.
# They might need adjustment if the script is moved.
BASE_SOURCE_DIR="/mnt/ecg_project/data/processed/for_best_resample"
BASE_DEST_DIR="/mnt/ecg_project/data/processed/15ch_arrange_direction"

echo "Base Source Directory: $BASE_SOURCE_DIR"
echo "Base Destination Directory: $BASE_DEST_DIR"
echo "---"

# Group 1: Subjects to use '15ch_diff_from_ch_1'
GROUP_1=("asano" "gosha" "ikejima" "kanda" "nakashimizu" "nishio" "noda" "takahashi_jr")

# Group 2: Subjects to use '15ch_diff_from_ch_16'
GROUP_2=("goto" "kawai" "matumoto" "patient4" "patient6" "patient8" "patient9" "taniguchi" "yoshikura")

# --- Process Group 1 ---
echo "Processing Group 1 (using 15ch_diff_from_ch_1)..."
for subject in "${GROUP_1[@]}"; do
    # Use a wildcard (*) to match any date/suffix combination (e.g., asano_0714_0.8s)
    SOURCE_SUBDIR_PATTERN="${subject}_*"
    
    # Find the source directory matching the pattern
    # Note: This assumes there is only one matching directory per subject.
    SOURCE_PATH_CANDIDATE=(${BASE_SOURCE_DIR}/${SOURCE_SUBDIR_PATTERN}/15ch_diff_from_ch_1/)
    
    if [ -d "${SOURCE_PATH_CANDIDATE[0]}" ]; then
        SOURCE_PATH="${SOURCE_PATH_CANDIDATE[0]}"
        DEST_PATH_DIR=$(dirname "${SOURCE_PATH_CANDIDATE[0]}") # e.g., /.../asano_0714_0.8s
        DEST_SUBDIR=$(basename "$DEST_PATH_DIR") # e.g., asano_0714_0.8s
        DEST_PATH="${BASE_DEST_DIR}/${DEST_SUBDIR}/0/"

        echo "Subject: $subject"
        echo "  Source:      $SOURCE_PATH"
        echo "  Destination: $DEST_PATH"
        
        # Create destination directory
        mkdir -p "$DEST_PATH"
        
        # Copy all files and directories recursively
        cp -rv "$SOURCE_PATH"* "$DEST_PATH"
        echo "  ...Done."
    else
        echo "  WARNING: No source directory found for subject pattern: ${subject}_*"
    fi
    echo ""
done

# --- Process Group 2 ---
echo "Processing Group 2 (using 15ch_diff_from_ch_16)..."
for subject in "${GROUP_2[@]}"; do
    SOURCE_SUBDIR_PATTERN="${subject}_*"
    SOURCE_PATH_CANDIDATE=(${BASE_SOURCE_DIR}/${SOURCE_SUBDIR_PATTERN}/15ch_diff_from_ch_16/)

    if [ -d "${SOURCE_PATH_CANDIDATE[0]}" ]; then
        SOURCE_PATH="${SOURCE_PATH_CANDIDATE[0]}"
        DEST_PATH_DIR=$(dirname "${SOURCE_PATH_CANDIDATE[0]}")
        DEST_SUBDIR=$(basename "$DEST_PATH_DIR")
        DEST_PATH="${BASE_DEST_DIR}/${DEST_SUBDIR}/0/"

        echo "Subject: $subject"
        echo "  Source:      $SOURCE_PATH"
        echo "  Destination: $DEST_PATH"
        
        mkdir -p "$DEST_PATH"
        
        cp -rv "$SOURCE_PATH"* "$DEST_PATH"
        echo "  ...Done."
    else
        echo "  WARNING: No source directory found for subject pattern: ${subject}_*"
    fi
    echo ""
done

echo "--- Script finished. ---"
