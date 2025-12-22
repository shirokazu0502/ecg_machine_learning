import torch
from torch.utils.data import DataLoader
import os
import random
from glob import glob
import numpy as np
import pandas as pd
import itertools
import datetime

import utils
from models import CNN_LSTM_Model
import arguments as config
import Dataset

# --- Configuration ---
ALL_TARGET_LEADS = ["A1", "A2", "V1", "V2", "V3", "V4", "V5", "V6"]
DATASET_NAME = "15ch_arrange_direction"
GRID_SEARCH_EPOCHS = 500  # Epochs for each grid search trial

PARAM_GRID = {
    "learning_rate": [0.001],
    "train_batch_size": [4, 8, 16],
    "lstm_hidden_size": [64, 128],
    "cnn_filters_2": [64, 128],
    "lstm_num_layers": [1, 2],
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


def find_best_params_for_lead(cli_args, all_subjects, target_lead):
    """
    Phase 1: Finds the best hyperparameters for a single lead.
    """
    print(f"\n--- Phase 1: Finding Best Params for Lead: {target_lead} ---")

    patient_subjects = [s for s in all_subjects if "patient" in s]
    other_subjects = [s for s in all_subjects if "patient" not in s]

    if not patient_subjects or len(other_subjects) < 2:
        raise ValueError(
            "Not enough 'patient' and 'other' subjects to create the evaluation set."
        )

    val_set_p = random.sample(patient_subjects, 1)
    val_set_o = random.sample(other_subjects, 2)
    evaluation_set = val_set_p + val_set_o
    random.shuffle(evaluation_set)

    print(f"Using Evaluation Set for tuning: {evaluation_set}")

    keys, values = zip(*PARAM_GRID.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    best_score = -1
    best_params = None

    for i, params in enumerate(param_combinations):
        print(f"\n- Evaluating Combo {i+1}/{len(param_combinations)}: {params}")

        fold_scores = []
        for j, val_subject in enumerate(evaluation_set):
            train_subjects = [s for s in evaluation_set if s != val_subject]
            print(f"  - Fold {j+1}/3: Validating on '{val_subject}'")

            common_dataset_args = {
                "Dataset_name": DATASET_NAME,
                "dataset_num": cli_args.dataset_num,
                "DataAugmentation": cli_args.DataAugmentation,
                "ave_data_flg": cli_args.ave_data_flg,
                "num_channels": cli_args.num_channels,
                "target_lead": target_lead,
            }
            train_ds = Dataset.Dataset_setup_8ch_pt_augmentation(
                include_subjects=train_subjects, **common_dataset_args
            )
            val_ds = Dataset.Dataset_setup_8ch_pt_augmentation(
                include_subjects=[val_subject], **common_dataset_args
            )

            train_loader = DataLoader(
                train_ds, batch_size=params["train_batch_size"], shuffle=True
            )
            val_loader = DataLoader(
                val_ds, batch_size=params["train_batch_size"], shuffle=False
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

            for _ in range(GRID_SEARCH_EPOCHS):
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

            pearson_corr, _ = utils.pearsonr(
                torch.cat(all_recon_x).flatten().numpy(),
                torch.cat(all_xo).flatten().numpy(),
            )
            fold_scores.append(pearson_corr)

        avg_score = np.mean(fold_scores)
        print(f"  - Avg Pearson for combo: {avg_score:.6f}")

        if avg_score > best_score:
            best_score = avg_score
            best_params = params
            print("  *** New best score for this lead! ***")

    print(
        f"--- Best Params for {target_lead}: {best_params} (Score: {best_score:.6f}) ---"
    )
    return best_params


def evaluate_loocv(best_params_per_lead, all_subjects, cli_args):
    """
    Phase 2: Performs a full Leave-One-Out Cross-Validation on ALL subjects
    using the best hyperparameters found for each lead.
    """
    print("\n\n--- Phase 2: Final Evaluation using Full LOOCV ---")

    all_fold_results = []

    for i, test_subject in enumerate(all_subjects):
        print(
            f"\n- LOOCV Fold {i+1}/{len(all_subjects)}: Testing on '{test_subject}' -"
        )

        training_subjects = [s for s in all_subjects if s != test_subject]
        trained_models = {}
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Train 8 models for this fold, each with its best params
        for lead in ALL_TARGET_LEADS:
            print(f"  - Training model for lead: {lead}")
            params = best_params_per_lead[lead]

            common_dataset_args = {
                "Dataset_name": DATASET_NAME,
                "dataset_num": cli_args.dataset_num,
                "DataAugmentation": cli_args.DataAugmentation,
                "ave_data_flg": cli_args.ave_data_flg,
                "num_channels": cli_args.num_channels,
                "target_lead": lead,
            }
            train_ds = Dataset.Dataset_setup_8ch_pt_augmentation(
                include_subjects=training_subjects, **common_dataset_args
            )
            train_loader = DataLoader(
                train_ds, batch_size=params["train_batch_size"], shuffle=True
            )

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

            for epoch in range(cli_args.epochs):  # Use full epochs for final model
                model.train()
                for x, xo, _, _ in train_loader:
                    x, xo = x.to(device), xo.to(device)
                    optimizer.zero_grad()
                    outputs = model(x)
                    loss = criterion(outputs, xo)
                    loss.backward()
                    optimizer.step()

            trained_models[lead] = model

        # Evaluate the ensemble on the held-out test subject
        print(f"  - Evaluating on test subject: {test_subject}")
        test_ds_args = {
            "Dataset_name": DATASET_NAME,
            "dataset_num": cli_args.dataset_num,
            "DataAugmentation": "",
            "ave_data_flg": cli_args.ave_data_flg,
            "num_channels": cli_args.num_channels,
            "target_lead": None,
        }
        test_dataset = Dataset.Dataset_setup_8ch_pt_augmentation(
            include_subjects=[test_subject], **test_ds_args
        )
        test_loader = DataLoader(
            test_dataset, batch_size=cli_args.train_batch_size, shuffle=False
        )

        all_recon_x, all_xo_full = [], []
        with torch.no_grad():
            for x, xo_full, _, _ in test_loader:
                x = x.to(device)
                recon_x_leads = [trained_models[lead](x) for lead in ALL_TARGET_LEADS]
                all_recon_x.append(torch.cat(recon_x_leads, dim=1).cpu())
                all_xo_full.append(xo_full.cpu())

        recon_x_all = torch.cat(all_recon_x)
        xo_all = torch.cat(all_xo_full)

        pearson_all, _ = utils.pearsonr(
            recon_x_all.flatten().numpy(), xo_all.flatten().numpy()
        )
        mae_loss = utils.MAE_2(reduction="none")(recon_x_all, xo_all)
        mae_per_channel = np.mean(
            [item.tolist() for item in utils.cul_val_per_12ch_no_pt(mae_loss)], axis=0
        )

        fold_metrics = {"fold_subject": test_subject, "pearson_all": pearson_all}
        for j, lead in enumerate(ALL_TARGET_LEADS):
            fold_metrics[f"mae_{lead}"] = mae_per_channel[j]

        all_fold_results.append(fold_metrics)
        print(f"  - Fold {i+1} Pearson: {fold_metrics['pearson_all']:.6f}")

    # Aggregate and save final results
    results_df = pd.DataFrame(all_fold_results)
    final_avg_metrics = results_df.mean(numeric_only=True)

    print("\n\n--- LOOCV Final Results (Averaged Across All Folds) ---")
    print(final_avg_metrics)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_df.to_csv(f"final_loocv_results_{timestamp}.csv", index=False)
    final_avg_metrics.to_csv(
        f"final_loocv_average_{timestamp}.csv", header=["average_value"]
    )
    print(f"\nSaved detailed LOOCV results to: final_loocv_results_{timestamp}.csv")


def main():
    cli_args = config.get_args()
    all_subjects = get_all_subject_names(DATASET_NAME)

    # Phase 1: Find best params for each lead
    best_params_per_lead = {}
    for lead in ALL_TARGET_LEADS:
        best_params = find_best_params_for_lead(cli_args, all_subjects, lead)
        if best_params is None:
            print(
                f"Warning: Grid search failed for lead {lead}. Skipping final evaluation for this lead."
            )
            # Use default params as a fallback
            best_params_per_lead[lead] = {
                "learning_rate": 0.001,
                "train_batch_size": 16,
                "lstm_hidden_size": 128,
                "cnn_filters_2": 64,
                "lstm_num_layers": 1,
            }
        else:
            best_params_per_lead[lead] = best_params

    print("\n\n--- Finished Hyperparameter Tuning for All Leads ---")
    for lead, params in best_params_per_lead.items():
        print(f"  - Best for {lead}: {params}")

    # Phase 2: Run final evaluation using the best parameters found
    evaluate_loocv(best_params_per_lead, all_subjects, cli_args)


if __name__ == "__main__":
    main()
