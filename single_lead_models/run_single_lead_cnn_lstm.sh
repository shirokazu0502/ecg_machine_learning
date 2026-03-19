#!/bin/bash

# List of unique TARGET_NAME values
TARGET_NAMES=("patient4" "asano" "gosha" "goto" "ikejima" "nishio" "takahashi_jr" "kanda" "kawai" "matumoto" "nakashimizu" "noda" "taniguchi" "yoshikura"  "patient6" "patient8" "patient9")

# Define common arguments
COMMON_ARGS=" \
    --Dataset_name 15ch_arrange_direction \
    --num_channels 15 \
    --ave_data_flg 1 \
    --cnn_filters_1 32 \
    --cnn_filters_2 64 \
    --cnn_kernel_size 5 \
    --lstm_hidden_size 128 \
    --lstm_num_layers 1 \
    --epochs 1000
"
# --DataAugmentation 'st_warp,t_height' \


# Note: epochs reduced for quick test. Change back to 1000 for full training.

for name in "${TARGET_NAMES[@]}"; do
    echo "Running for TARGET_NAME: $name"

    # Training for all leads
    python3 train_single_lead_cnn_lstm.py \
        --TARGET_NAME "$name" \
        --mode train \
        --train_batch_size 16 \
        $COMMON_ARGS
    
    # Testing for all leads (after all individual models are trained)
    # The test mode will load all 8 models and aggregate results
    python3 train_single_lead_cnn_lstm.py \
        --TARGET_NAME "$name" \
        --mode test \
        $COMMON_ARGS

done

echo "Single-lead model training and testing script finished."

# --- Aggregation Step ---
echo -e "\n--- Aggregating all result CSVs into a single file ---"

# Default MAE folder from arguments.py
MAE_DIR="output/mae"
DATASET_NAME="15ch_arrange_direction" # Should match the one used in the script
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SUMMARY_FILE="${MAE_DIR}/MAE_summary_single_lead_${DATASET_NAME}_${TIMESTAMP}.csv"

# Find all the individual CSV files generated in this run
# Note: This looks for files with the specific name format in the default directory
CSV_FILES=$(find "$MAE_DIR" -name "MAE_single_lead_ensemble_${DATASET_NAME}_*.csv")

if [ -z "$CSV_FILES" ]; then
    echo "Warning: No individual result CSVs found to aggregate in $MAE_DIR."
else
    # Create the directory for the summary file if it doesn't exist
    mkdir -p "$MAE_DIR"
    
    # Get header from the first file and create/overwrite the summary file
    head -n 1 $(echo "$CSV_FILES" | head -n 1) > "$SUMMARY_FILE"
    
    # Append the content (data rows) of all found CSV files to the summary file
    for f in $CSV_FILES; do
        tail -n +2 "$f" >> "$SUMMARY_FILE"
    done
    
    echo "Successfully aggregated results into: $SUMMARY_FILE"
fi

