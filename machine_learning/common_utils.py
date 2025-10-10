# common_utils.py

import torch
import numpy as np
import os
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from vae_model import loss_function_multi_head


# --- データローダ作成 ---
def create_dataloaders_for_fold(
    full_dataset, subject_ids, train_subjects, test_subject_id, batch_size, val_size=0.2
):
    """
    特定のfold（ループ）のための訓練・検証・テスト用DataLoaderを作成する。
    """
    train_fold_subjects, val_fold_subjects = train_test_split(
        train_subjects, test_size=val_size, random_state=42
    )

    train_indices = [
        i for i, sub_id in enumerate(subject_ids) if sub_id in train_fold_subjects
    ]
    val_indices = [
        i for i, sub_id in enumerate(subject_ids) if sub_id in val_fold_subjects
    ]
    test_indices = [
        i for i, sub_id in enumerate(subject_ids) if sub_id == test_subject_id
    ]

    train_dataset_fold = Subset(full_dataset, train_indices)
    val_dataset_fold = Subset(full_dataset, val_indices)
    test_dataset_fold = Subset(full_dataset, test_indices)

    train_loader = DataLoader(
        train_dataset_fold,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset_fold,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset_fold,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


# --- MultiHeadVAE専用の訓練・評価関数 ---
def train_multi_head_vae(model, dataloader, optimizer, device, beta):
    model.train()
    total_loss, total_recon_loss, total_kld = 0, 0, 0
    for batch in tqdm(dataloader, desc="Training VAE", leave=False):
        inputs, targets, _, _ = batch  # pt_index and label_name are ignored here
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        recon_p, recon_r, recon_t, mu, logvar = model(inputs)
        loss, recon_loss, kld = loss_function_multi_head(
            recon_p, recon_r, recon_t, targets, mu, logvar, beta=beta
        )
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_recon_loss += recon_loss.item()
        total_kld += kld.item()
    n = len(dataloader)
    return total_loss / n, total_recon_loss / n, total_kld / n


def evaluate_multi_head_vae(model, dataloader, device, beta):
    model.eval()
    total_loss, total_recon_loss, total_kld = 0, 0, 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating VAE", leave=False):
            inputs, targets, _, _ = batch  # pt_index and label_name are ignored here
            inputs, targets = inputs.to(device), targets.to(device)
            recon_p, recon_r, recon_t, mu, logvar = model(inputs)
            loss, recon_loss, kld = loss_function_multi_head(
                recon_p, recon_r, recon_t, targets, mu, logvar, beta=beta
            )
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kld += kld.item()
    n = len(dataloader)
    return total_loss / n, total_recon_loss / n, total_kld / n


# --- Early Stopping クラス ---
class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path="checkpoint.pt"):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model to {self.path}"
            )
        torch.save(model.state_dict(), self.path)
