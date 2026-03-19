import torch
from torch.utils.data import DataLoader
import os
import random
from glob import glob
import numpy as np
import pandas as pd
import datetime

import utils
from models import CNN_LSTM_Model
import arguments as config
import Dataset

# --- Configuration ---
# ==============================================================================
#  [!] ACTION REQUIRED: Please update this dictionary with the best parameters
#      found by the grid_search.py script.
# ==============================================================================
BEST_PARAMS = {
    "learning_rate": 0.001,
    "train_batch_size": 16,
    "lstm_hidden_size": 128,
    "cnn_filters_2": 64,
    "lstm_num_layers": 2,
}

# All leads to be trained and evaluated
ALL_TARGET_LEADS = ["A1", "A2", "V1", "V2", "V3", "V4", "V5", "V6"]
DATASET_NAME = "15ch_arrange_direction"


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


def train_and_evaluate_fold(test_subject, all_subjects, params, cli_args):
    """
    Performs a full Leave-One-Out fold. Trains 8 models on N-1 subjects
    and evaluates them on the single held-out test_subject.
    """
    training_subjects = [s for s in all_subjects if s != test_subject]
    trained_models = {}

    # --- 1. Train a model for each lead ---
    for lead in ALL_TARGET_LEADS:
        print(f"    - Training model for lead: {lead}")

        common_dataset_args = {
            "Dataset_name": DATASET_NAME,
            "dataset_num": cli_args.dataset_num,
            "DataAugmentation": cli_args.DataAugmentation,
            "ave_data_flg": cli_args.ave_data_flg,
            "num_channels": cli_args.num_channels,
            "target_lead": lead,
        }
        train_dataset = Dataset.Dataset_setup_8ch_pt_augmentation(
            include_subjects=training_subjects, **common_dataset_args
        )
        train_loader = DataLoader(
            train_dataset, batch_size=params["train_batch_size"], shuffle=True
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_args = {
            "num_channels": cli_args.num_channels,
            "out_channels": 1,
            "cnn_filters_1": 32,
            "cnn_filters_2": params["cnn_filters_2"],
            "cnn_kernel_size": 5,
            "lstm_hidden_size": params["lstm_hidden_size"],
            "lstm_num_layers": params["lstm_num_layers"],
        }
        model = CNN_LSTM_Model(**model_args).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])
        criterion = utils.loss_fn_lstm

        for epoch in range(cli_args.epochs):  # Using full epochs
            model.train()
            for x, xo, _, _ in train_loader:
                x, xo = x.to(device), xo.to(device)
                optimizer.zero_grad()
                outputs = model(x)
                loss = criterion(outputs, xo)
                loss.backward()
                optimizer.step()

        trained_models[lead] = model

    # --- 2. Evaluate the ensemble of models on the test subject ---
    print(f"    - Evaluating on test subject: {test_subject}")

    test_dataset_args = {
        "Dataset_name": DATASET_NAME,
        "dataset_num": cli_args.dataset_num,
        "DataAugmentation": "",
        "ave_data_flg": cli_args.ave_data_flg,
        "num_channels": cli_args.num_channels,
        "target_lead": None,  # Get all 8 leads for ground truth
    }
    test_dataset = Dataset.Dataset_setup_8ch_pt_augmentation(
        include_subjects=[test_subject], **test_dataset_args
    )
    test_loader = DataLoader(
        test_dataset, batch_size=params["train_batch_size"], shuffle=False
    )

    # --- Perform inference and stitch results ---
    all_recon_x = []
    all_xo_full = []
    with torch.no_grad():
        for x, xo_full, _, _ in test_loader:
            x = x.to(device)
            recon_x_leads = []
            for lead in ALL_TARGET_LEADS:
                recon_x_single = trained_models[lead](x)
                recon_x_leads.append(recon_x_single)

            recon_x = torch.cat(recon_x_leads, dim=1)
            all_recon_x.append(recon_x.cpu())
            all_xo_full.append(xo_full.cpu())

    recon_x_all = torch.cat(all_recon_x)
    xo_all = torch.cat(all_xo_full)

    # --- 3. Calculate and return metrics for this fold ---
    pearson_all, _ = utils.pearsonr(
        recon_x_all.flatten().numpy(), xo_all.flatten().numpy()
    )

    mae_loss = utils.MAE_2(reduction="none")(recon_x_all, xo_all)
    mae_per_channel = np.mean(
        [item.tolist() for item in utils.cul_val_per_12ch_no_pt(mae_loss)], axis=0
    )

    fold_metrics = {"fold_subject": test_subject, "pearson_all": pearson_all}
    for i, lead in enumerate(ALL_TARGET_LEADS):
        fold_metrics[f"mae_{lead}"] = mae_per_channel[i]

    return fold_metrics


def main():
    cli_args = config.get_args()
    all_subjects = get_all_subject_names(DATASET_NAME)

    print("--- Final Model Evaluation using Leave-One-Out Cross-Validation ---")
    print(f"Using Hyperparameters: {BEST_PARAMS}")
    print(f"Total subjects: {len(all_subjects)}")
    print(f"This will perform {len(all_subjects)} full training and evaluation runs.")

    all_results = []

    # Main LOOCV loop
    for i, test_subject in enumerate(all_subjects):
        print(
            f"\n--- Starting Fold {i+1}/{len(all_subjects)}: Test Subject = {test_subject} ---"
        )

        fold_result = train_and_evaluate_fold(
            test_subject, all_subjects, BEST_PARAMS, cli_args
        )
        all_results.append(fold_result)
        print(f"--- Finished Fold {i+1}. Pearson: {fold_result['pearson_all']:.6f} ---")

    # --- Aggregate and Save Final Results ---
    results_df = pd.DataFrame(all_results)

    # Calculate and print the mean of all metrics
    final_avg_metrics = results_df.mean(numeric_only=True)
    print("\n\n--- LOOCV Final Results (Averaged Across All Folds) ---")
    print(final_avg_metrics)

    # Save the detailed and averaged results
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_df.to_csv(f"final_loocv_results_{timestamp}.csv", index=False)
    final_avg_metrics.to_csv(
        f"final_loocv_average_{timestamp}.csv", header=["average_value"]
    )

    print(f"\nSaved detailed results to: final_loocv_results_{timestamp}.csv")
    print(f"Saved averaged results to: final_loocv_average_{timestamp}.csv")
    print("--- Evaluation Complete ---")


if __name__ == "__main__":
    main()
