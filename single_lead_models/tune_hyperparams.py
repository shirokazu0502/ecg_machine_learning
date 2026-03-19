import optuna
import torch
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import os
import random
from glob import glob
import numpy as np

import utils
from models import CNN_LSTM_Model
import arguments as config
import Dataset

# --- Constants ---
# Define the lead we are optimizing for
OPTIMIZE_LEAD = "A2"
# Number of trials Optuna will run
N_TRIALS = 25
# Epochs for each trial (keep it low for speed)
TUNE_EPOCHS = 10 
# Dataset name
DATASET_NAME = "15ch_arrange_direction"


def get_all_subject_names(dataset_name):
    """
    Scans the data directory to get a list of unique subject names.
    e.g., extracts 'asano', 'gosha' from 'asano_0714_0.8s', 'gosha_0807_0.8s'
    """
    base_dir = f"/mnt/ecg_project/data/processed/{dataset_name}"
    subject_dirs = [os.path.basename(os.path.normpath(d)) for d in glob(f"{base_dir}/*/") if os.path.isdir(d)]
    # Extract the name part before the first underscore
    subject_names = sorted(list(set([d.split('_')[0] for d in subject_dirs])))
    return subject_names

def objective(trial: optuna.Trial):
    """
    The main objective function for Optuna to optimize.
    A single "trial" represents one full training and validation run with a specific set of hyperparameters.
    """
    # --- 1. Suggest Hyperparameters ---
    # Define the search space for Optuna to explore.
    args = config.get_args()
    args.learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    args.train_batch_size = trial.suggest_categorical("train_batch_size", [8, 16, 32])
    args.cnn_filters_1 = trial.suggest_categorical("cnn_filters_1", [16, 32, 64])
    args.cnn_filters_2 = trial.suggest_categorical("cnn_filters_2", [32, 64, 128])
    args.lstm_hidden_size = trial.suggest_categorical("lstm_hidden_size", [64, 128, 256])

    print(f"\n--- Starting Trial {trial.number} ---")
    print(f"  - LR: {args.learning_rate:.6f}")
    print(f"  - Batch Size: {args.train_batch_size}")
    print(f"  - CNN Filters: {args.cnn_filters_1}, {args.cnn_filters_2}")
    print(f"  - LSTM Hidden: {args.lstm_hidden_size}")

    # --- 2. Prepare Datasets (Leave-Two-Out) ---
    all_subjects = get_all_subject_names(DATASET_NAME)
    
    # For this study, let's fix the test subject for consistency across trials
    # and randomly choose a validation subject from the rest.
    test_subject = "asano" 
    
    train_val_candidates = [s for s in all_subjects if s != test_subject]
    validation_subject = random.choice(train_val_candidates)
    
    training_subjects = [s for s in train_val_candidates if s != validation_subject]

    print(f"  - Test Subject: {test_subject}")
    print(f"  - Validation Subject: {validation_subject}")
    print(f"  - Training Subjects: {len(training_subjects)} subjects")

    # Use the flexible dataset function to create our splits
    common_dataset_args = {
        "Dataset_name": DATASET_NAME,
        "dataset_num": args.dataset_num,
        "DataAugmentation": args.DataAugmentation,
        "ave_data_flg": args.ave_data_flg,
        "num_channels": args.num_channels,
        "target_lead": OPTIMIZE_LEAD,
    }
    
    train_dataset = Dataset.Dataset_setup_8ch_pt_augmentation(include_subjects=training_subjects, **common_dataset_args)
    val_dataset = Dataset.Dataset_setup_8ch_pt_augmentation(include_subjects=[validation_subject], **common_dataset_args)

    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.train_batch_size, shuffle=False)

    # --- 3. Train and Evaluate the Model ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN_LSTM_Model(num_channels=args.num_channels, out_channels=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = utils.loss_fn_lstm
    
    # Train for a few epochs
    for epoch in range(TUNE_EPOCHS):
        model.train()
        for x, xo, _, _ in train_loader:
            x, xo = x.to(device), xo.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, xo)
            loss.backward()
            optimizer.step()
        print(f"    Epoch {epoch+1}/{TUNE_EPOCHS} done.")

    # Evaluate on the validation set
    model.eval()
    all_recon_x = []
    all_xo = []
    with torch.no_grad():
        for x, xo, _, _ in val_loader:
            x = x.to(device)
            recon_x = model(x)
            all_recon_x.append(recon_x.cpu())
            all_xo.append(xo.cpu())

    recon_x_all = torch.cat(all_recon_x)
    xo_all = torch.cat(all_xo)

    # Calculate Pearson correlation, which we want to maximize
    pearson_corr, _ = utils.pearsonr(recon_x_all.flatten().numpy(), xo_all.flatten().numpy())
    
    print(f"  - Trial {trial.number} Finished. Validation Pearson: {pearson_corr:.6f}")
    
    # Optuna will maximize this value
    return pearson_corr


if __name__ == "__main__":
    # Create a new study
    study = optuna.create_study(direction="maximize")
    
    # Start the optimization
    study.optimize(objective, n_trials=N_TRIALS)
    
    # --- Print Results ---
    print("\n--- OPTIMIZATION FINISHED ---")
    print(f"Number of finished trials: {len(study.trials)}")

    print("\nBest trial:")
    best_trial = study.best_trial

    print(f"  - Value (Pearson Correlation): {best_trial.value:.6f}")

    print("  - Params: ")
    for key, value in best_trial.params.items():
        print(f"    - {key}: {value}")
        
    # --- Optional: Save results ---
    df = study.trials_dataframe()
    df.to_csv("tuning_results.csv", index=False)
    print("\nSaved tuning results to tuning_results.csv")
