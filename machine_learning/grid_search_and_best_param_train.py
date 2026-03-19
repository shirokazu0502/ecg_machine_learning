import torch
from torch.utils.data import DataLoader
import os
import random
from glob import glob
import numpy as np
import time
import itertools
import pandas as pd
import datetime
import subprocess
import sys  # Import sys

# Add the directory containing this script to sys.path
# This ensures that local modules like 'arguments' and 'Dataset' are imported correctly.
# script_dir = os.path.dirname(__file__)
# if script_dir not in sys.path:
#     sys.path.insert(0, script_dir)
# Add the project root to the Python path
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

import utils
import models
import arguments as config
import Dataset
from config.settings import (
    OUTPUT_DIR,
)

# --- Configuration ---
GRID_SEARCH_EPOCHS = 500
DATASET_NAME = "reference_center_4ch"

PARAM_GRIDS = {
    "cnn_lstm": {
        "learning_rate": [1e-4],
        "train_batch_size": [8, 16],
        "lstm_hidden_size": [64, 128],
        "cnn_filters_2": [64, 128],
    },
    # "cnn": {
    #     "learning_rate": [1e-4, 1e-3],
    #     "train_batch_size": [32, 64],
    #     "cnn_depth": [4],
    #     "cnn_init_filters": [16, 32],
    # },
    "cnn": {
        "learning_rate": [1e-4],
        "train_batch_size": [64],
        "cnn_depth": [4],
        "cnn_init_filters": [64],
    },
    "lstm": {
        "learning_rate": [1e-4],
        "train_batch_size": [64],
        "lstm_hidden_size": [256],
        "lstm_num_layers": [2],
    },
    "unet": {
        "learning_rate": [1e-4],
        "train_batch_size": [64],
        "base_filters": [32],
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
    # Handle names like 'takahashi_jr' by joining all parts except the last two (date and suffix)
    subject_names = sorted(
        list(set(["_".join(d.split("_")[:-2]) for d in subject_dirs]))
    )
    return subject_names


def evaluate_params_cv(params, all_subjects, validation_subjects, cli_args):
    """
    Evaluates a single set of hyperparameters using 3-run Leave-One-Out CV.
    Returns average Pearson correlation, MAE, and RMSE.
    """
    fold_pearson_scores = []
    fold_mae_scores = []
    fold_rmse_scores = []

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
            criterion = utils.loss_fn_weighted_mse_and_corr
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
            criterion = utils.loss_fn_mse_and_corr
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])

        for epoch in range(GRID_SEARCH_EPOCHS):
            model.train()
            for x, xo, _, pt_index in train_loader:  # Unpack pt_index
                x, xo, pt_index = (
                    x.to(device),
                    xo.to(device),
                    pt_index.to(device),
                )  # Move pt_index to device
                optimizer.zero_grad()
                outputs = model(x)
                if criterion == utils.loss_fn_weighted_mse_and_corr:
                    loss = criterion(
                        outputs,
                        xo,
                        pt_index,  # Pass pt_index
                        beta=cli_args.beta,  # Pass beta
                        weight_r=cli_args.r_loss_weight,
                        weight_other=cli_args.other_loss_weight,  # Corrected argument name
                    )
                else:
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

        # Calculate all three metrics
        pearson_corr, _ = utils.pearsonr(
            recon_x_all.flatten().numpy(), xo_all.flatten().numpy()
        )
        mae_score = torch.nn.functional.l1_loss(recon_x_all, xo_all).item()
        rmse_score = torch.sqrt(
            torch.nn.functional.mse_loss(recon_x_all, xo_all)
        ).item()

        fold_pearson_scores.append(pearson_corr)
        fold_mae_scores.append(mae_score)
        fold_rmse_scores.append(rmse_score)
        print(
            f"    - Fold {i+1} Pearson: {pearson_corr:.6f}, MAE: {mae_score:.6f}, RMSE: {rmse_score:.6f}"
        )

    avg_pearson = np.mean(fold_pearson_scores)
    avg_mae = np.mean(fold_mae_scores)
    avg_rmse = np.mean(fold_rmse_scores)

    print(
        f"  - Avg scores for this combo: Pearson: {avg_pearson:.6f}, MAE: {avg_mae:.6f}, RMSE: {avg_rmse:.6f}"
    )
    return avg_pearson, avg_mae, avg_rmse


def run_final_training(best_params, model_type, all_subjects, cli_args):
    """
    Phase 2: Runs the final training and evaluation for all subjects using the best params.
    """
    print("\n--- Phase 2: Starting Final Training & Evaluation for All Subjects ---")

    # Map model_type to the correct training script
    script_map = {
        "cnn_lstm": "train_cnn_lstm.py",
        "cnn": "train_cnn.py",
        "lstm": "train_lstm.py",
        "unet": "train_unet.py",
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
            "--model_type",
            model_type,  # Add this line
            "--Dataset_name",
            DATASET_NAME,
            "--num_channels",
            str(cli_args.num_channels),
            "--ave_data_flg",
            str(cli_args.ave_data_flg),
            "--DataAugmentation",
            cli_args.DataAugmentation,
            "--epochs",
            str(cli_args.epochs),
            "--beta",
            str(cli_args.beta),
            "--unet_depth",
            str(cli_args.unet_depth),
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
    cli_args.num_channels = 16  # データセットに合わせて15か16を指定
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

    # --- Create a unique directory for this grid search run ---
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H")
    results_dir = os.path.join(
        OUTPUT_DIR, "grid_search_results", f"{model_type}_{timestamp}"
    )
    print(results_dir)
    # もし存在しないならディレクトリを作成
    os.makedirs(results_dir, exist_ok=True)
    results_csv_path = os.path.join(results_dir, "grid_search_summary.csv")

    print(f"Grid search results will be saved to: {results_csv_path}")

    # --- Phase 1: Hyperparameter Search ---
    # random.seed(cli_args.seed) # No longer needed for fixed validation subjects
    # validation_subjects = random.sample(all_subjects, 3) # Commented out: Fixed validation subjects
    validation_subjects = [
        "asano",
        "patient4",
        "nakashimizu",
    ]

    print("--- Phase 1: Grid Search Hyperparameter Tuning ---")
    print(f"Model Type: {model_type}")
    print(f"Subjects for Cross-Validation: {validation_subjects}")

    param_grid = PARAM_GRIDS[model_type]
    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    print(f"Starting grid search across {len(param_combinations)} combinations...")

    all_grid_search_results = []
    best_score = -1  # Best score will be based on Pearson correlation
    # まずparam_combinationsの最初の要素をbest_paramsとして初期化
    best_params = param_combinations[0]

    # param_combinationsが一つ以上の場合にgrid searchを実行
    if len(param_combinations) > 1:
        for i, params in enumerate(param_combinations):
            print(
                f"\n- Evaluating Combination {i+1}/{len(param_combinations)}: {params}"
            )
            avg_pearson, avg_mae, avg_rmse = evaluate_params_cv(
                params, all_subjects, validation_subjects, cli_args
            )

            # Log results for this combination
            result_entry = {
                **params,
                "avg_pearson": avg_pearson,
                "avg_mae": avg_mae,
                "avg_rmse": avg_rmse,
            }
            all_grid_search_results.append(result_entry)

            if avg_pearson > best_score:
                best_score = avg_pearson
                best_params = params
                print(f"  *** New best score found! Avg Pearson: {best_score:.6f} ***")

        # --- Save all grid search results to CSV ---
        if all_grid_search_results:
            results_df = pd.DataFrame(all_grid_search_results)
            # Reorder columns to have metrics first
            metric_cols = ["avg_pearson", "avg_mae", "avg_rmse"]
            param_cols = [col for col in results_df.columns if col not in metric_cols]
            results_df = results_df[metric_cols + param_cols]
            results_df.sort_values(by="avg_pearson", ascending=False, inplace=True)
            results_df.to_csv(results_csv_path, index=False)
            print(f"\nGrid search results saved to {results_csv_path}")

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
