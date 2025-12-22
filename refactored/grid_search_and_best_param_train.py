import torch
from torch.utils.data import DataLoader
import os
import random
from glob import glob
import numpy as np
import itertools
import pandas as pd
import datetime
import subprocess
import sys  # Import sys

# Add the directory containing this script to sys.path
# This ensures that local modules like 'arguments' and 'Dataset' are imported correctly.
script_dir = os.path.dirname(__file__)
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

import utils
import models
import arguments as config
import Dataset

# --- Configuration ---
GRID_SEARCH_EPOCHS = 500
DATASET_NAME = "15ch_arrange_direction"

PARAM_GRIDS = {
    "cnn_lstm": {
        "learning_rate": [1e-4],
        "train_batch_size": [8, 16],
        "lstm_hidden_size": [64, 128],
        "cnn_filters_2": [64, 128],
    },
    "cnn": {
        "learning_rate": [1e-4, 1e-3],
        "train_batch_size": [16],
        "cnn_depth": [4],
        "cnn_init_filters": [8, 16],
    },
    "lstm": {
        "learning_rate": [1e-4],
        "train_batch_size": [16, 32],
        "lstm_hidden_size": [64, 128, 256],
        "lstm_num_layers": [1, 2],
    },
    "unet": {
        "learning_rate": [1e-4],
        "train_batch_size": [8, 16],
        "base_filters": [16, 32],
    },
}


def get_all_subject_names(dataset_name):
    """Gets a list of unique subject names from the data directory."""
    base_dir = f"/mnt/ecg_project/data/processed/{dataset_name}"
    subject_dirs = [
        os.path.basename(os.path.normpath(d))
        for d in glob(f"{base_dir}/*/")
        if os.path.isdir(d)
    ]
    subject_names = sorted(list(set([d.split("_")[0] for d in subject_dirs])))
    return subject_names


def evaluate_params_cv(params, all_subjects, validation_subjects, cli_args):
    """
    Evaluates a single set of hyperparameters using 3-run Leave-One-Out CV.
    """
    fold_scores = []

    for i, val_subject in enumerate(validation_subjects):
        print(
            f"    - Fold {i+1}/{len(validation_subjects)}: Validating on '{val_subject}'"
        )

        train_dataset, val_dataset = Dataset.Dataset_setup_8ch_pt_augmentation(
            TARGET_NAME=val_subject,
            Dataset_name=DATASET_NAME,
            dataset_num=cli_args.dataset_num,
            DataAugmentation=cli_args.DataAugmentation,
            ave_data_flg=cli_args.ave_data_flg,
            num_channels=cli_args.num_channels,
        )

        train_loader = DataLoader(
            train_dataset, batch_size=params["train_batch_size"], shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=params["train_batch_size"], shuffle=False
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --- Instantiate the correct model based on model_type ---
        model_type = cli_args.model_type
        if model_type == "cnn_lstm":
            model_args = {
                "num_channels": cli_args.num_channels,
                "out_channels": 8,
                "cnn_filters_1": 32,
                "cnn_filters_2": params["cnn_filters_2"],
                "cnn_kernel_size": 5,
                "lstm_hidden_size": params["lstm_hidden_size"],
                "lstm_num_layers": params.get("lstm_num_layers", 1),
            }
            model = models.CNN_LSTM_Model(**model_args).to(device)
            criterion = utils.loss_fn_lstm
        elif model_type == "cnn":
            model_args = {
                "num_channels": cli_args.num_channels,
                "out_channels": 8,
                "depth": params["cnn_depth"],
                "init_filters": params["cnn_init_filters"],
            }
            model = models.SimpleCNN(**model_args).to(device)
            criterion = utils.loss_fn_unet
        elif model_type == "lstm":
            model_args = {
                "input_size": cli_args.num_channels,
                "hidden_size": params["lstm_hidden_size"],
                "num_layers": params["lstm_num_layers"],
                "output_size": 8,
                "datalength": cli_args.datalength,
            }
            model = models.LSTMModel(**model_args).to(device)
            criterion = utils.loss_fn_lstm
        elif model_type == "unet":
            model_args = {
                "num_channels": cli_args.num_channels,
                "out_channels": 8,
                "base_filters": params["base_filters"],
            }
            model = models.UNet1D(**model_args).to(device)
            criterion = utils.loss_fn_unet
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])

        for epoch in range(GRID_SEARCH_EPOCHS):
            model.train()
            for x, xo, _, _ in train_loader:
                x, xo = x.to(device), xo.to(device)
                optimizer.zero_grad()
                outputs = model(x)
                loss = criterion(outputs, xo)
                loss.backward()
                optimizer.step()

        model.eval()
        all_recon_x, all_xo = [], []
        with torch.no_grad():
            for x, xo, _, _ in val_loader:
                all_recon_x.append(model(x.to(device)).cpu())
                all_xo.append(xo.cpu())

        recon_x_all = torch.cat(all_recon_x)
        xo_all = torch.cat(all_xo)

        pearson_corr, _ = utils.pearsonr(
            recon_x_all.flatten().numpy(), xo_all.flatten().numpy()
        )
        fold_scores.append(pearson_corr)
        print(f"    - Fold {i+1} Pearson: {pearson_corr:.6f}")

    avg_score = np.mean(fold_scores)
    print(f"  - Avg Pearson for this combo: {avg_score:.6f}")
    return avg_score


def run_final_training(best_params, model_type, all_subjects, cli_args):
    """
    Phase 2: Runs the final training and evaluation for all subjects using the best params.
    """
    print("\n--- Phase 2: Starting Final Training & Evaluation for All Subjects ---")

    # Map model_type to the correct training script
    script_map = {
        "cnn_lstm": "refactored/train_cnn_lstm.py",
        "cnn": "refactored/train_cnn.py",
        "lstm": "refactored/train_lstm.py",
        "unet": "refactored/train_unet.py",
    }
    train_script = script_map.get(model_type)
    if not train_script:
        print(f"Error: No training script mapped for model_type '{model_type}'")
        return

    # Convert best_params dict to command line arguments
    param_args = []
    for key, value in best_params.items():
        param_args.append(f"--{key}")
        param_args.append(str(value))

    # Loop through all subjects, holding each one out for testing
    for target_name in all_subjects:
        print(f"\n--- Running LOOCV for TARGET_NAME: {target_name} ---")

        # Base command arguments
        base_cmd = [
            "python3",
            train_script,
            "--TARGET_NAME",
            target_name,
            "--Dataset_name",
            DATASET_NAME,
            "--num_channels",
            str(cli_args.num_channels),
            "--ave_data_flg",
            str(cli_args.ave_data_flg),
            "--DataAugmentation",
            cli_args.DataAugmentation,
            "--epochs",
            str(cli_args.epochs),  # Use full epochs
        ]

        # Add the best hyperparameters
        final_cmd = base_cmd + param_args

        # --- Run Training ---
        print(f"Executing Training for {target_name}...")
        train_cmd = final_cmd + ["--mode", "train"]
        subprocess.run(train_cmd, check=True)

        # --- Run Testing ---
        print(f"Executing Testing for {target_name}...")
        test_cmd = final_cmd + ["--mode", "test"]
        subprocess.run(test_cmd, check=True)

    print(f"\n--- Finished Final Training and Evaluation for all subjects ---")


def main():
    cli_args = config.get_args()
    all_subjects = get_all_subject_names(DATASET_NAME)
    model_type = cli_args.model_type

    if model_type not in PARAM_GRIDS:
        print(
            f"Error: Unknown model_type '{model_type}'. Available types: {list(PARAM_GRIDS.keys())}"
        )
        return

    if len(all_subjects) < 3:
        print("Error: Need at least 3 subjects for cross-validation.")
        return

    # --- Phase 1: Hyperparameter Search ---
    random.seed(cli_args.seed)
    validation_subjects = random.sample(all_subjects, 3)

    print("--- Phase 1: Grid Search Hyperparameter Tuning ---")
    print(f"Model Type: {model_type}")
    print(f"Subjects for Cross-Validation: {validation_subjects}")

    param_grid = PARAM_GRIDS[model_type]
    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    print(f"Starting grid search across {len(param_combinations)} combinations...")

    best_score = -1
    best_params = None

    for i, params in enumerate(param_combinations):
        print(f"\n- Evaluating Combination {i+1}/{len(param_combinations)}: {params}")
        avg_pearson_score = evaluate_params_cv(
            params, all_subjects, validation_subjects, cli_args
        )
        if avg_pearson_score > best_score:
            best_score = avg_pearson_score
            best_params = params
            print(f"  *** New best score found! Avg Pearson: {best_score:.6f} ***")

    print("\n--- GRID SEARCH FINISHED ---")
    if best_params:
        print(f"Best Pearson Correlation (avg over 3 folds): {best_score:.6f}")
        print("Best Hyperparameters:")
        for key, value in best_params.items():
            print(f"  - {key}: {value}")
    else:
        print("Grid search did not find any valid parameters.")
        return

    # --- Phase 2: Final Training and Evaluation ---
    run_final_training(best_params, model_type, all_subjects, cli_args)


if __name__ == "__main__":
    main()
