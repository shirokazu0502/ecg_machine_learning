# main.py

import argparse
import torch
import numpy as np
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import random_split, DataLoader

# 必要な自作モジュールをインポート
import Dataset
from common_utils import train_multi_head_vae, evaluate_multi_head_vae, EarlyStopping
from vae_model import MultiHeadVAE


def get_subject_list(dataset_name):
    """
    データセットのディレクトリ構造から被験者リストを生成する。
    """
    try:
        directory_path = os.path.join(Dataset.PROCESSED_DATA_DIR, dataset_name)
        dir_names = Dataset.get_directory_names_all(directory_path)
        subjects = set()
        for name in dir_names:
            subject = name.split("/")[0] if "/" in name else name.split("\\")[0]
            subjects.add(subject)
        if not subjects:
            raise FileNotFoundError
        print(f"Found subjects: {sorted(list(subjects))}")
        return sorted(list(subjects))
    except (FileNotFoundError, AttributeError) as e:
        print(f"Error: Could not find directories in {directory_path}.")
        print(
            "Please check the 'Dataset_name' argument and your data directory structure."
        )
        return None


def main():
    parser = argparse.ArgumentParser(
        description="VAE ECG Reconstruction Training with LOO CV"
    )

    # --- Dataset.pyで必要な引数 ---
    parser.add_argument(
        "--Dataset_name", type=str, required=True, help="Name of the dataset directory"
    )
    parser.add_argument(
        "--p_augumentation",
        type=str,
        default="",
        help="Augmentation type for P-wave model",
    )
    parser.add_argument(
        "--r_augumentation",
        type=str,
        default="",
        help="Augmentation type for R-wave model",
    )
    parser.add_argument(
        "--t_augumentation",
        type=str,
        default="",
        help="Augmentation type for T-wave model",
    )
    parser.add_argument(
        "--transform_type",
        type=str,
        default="normal",
        help="Transform type ('normal', 'random')",
    )
    parser.add_argument(
        "--dataset_num", type=int, default=100, help="Number of CSV files per subject"
    )
    parser.add_argument(
        "--ave_data_flg", type=int, default=0, help="Flag to use average heartbeat data"
    )

    # --- モデルと学習の引数 ---
    parser.add_argument(
        "--epochs", type=int, default=200, help="Max number of training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for training"
    )
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument(
        "--patience", type=int, default=15, help="Patience for Early Stopping"
    )
    parser.add_argument(
        "--beta", type=float, default=1.0, help="Weight of the KLD term in VAE loss"
    )
    parser.add_argument(
        "--latent_dim", type=int, default=16, help="Dimension of the latent space"
    )
    parser.add_argument(
        "--data_len", type=int, default=400, help="Length of the ECG data sequence"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- リーブワンアウトでループするための被験者リストを取得 ---
    unique_subjects = get_subject_list(args.Dataset_name)
    if unique_subjects is None:
        return

    n_subjects = len(unique_subjects)
    fold_test_losses = []

    # ==========================================================
    #       リーブワンアウト交差検証のメインループ
    # ==========================================================
    for fold, target_name in enumerate(unique_subjects):

        print(
            f"\n{'='*20} Fold {fold+1}/{n_subjects} | Testing on Subject: {target_name} {'='*20}"
        )

        # --- このFold用のデータセットを Dataset.py を使って作成 ---
        train_dataset, test_dataset = Dataset.Dataset_setup_8ch_pt_augmentation(
            TARGET_NAME=target_name,
            transform_type=args.transform_type,
            Dataset_name=args.Dataset_name,
            dataset_num=args.dataset_num,
            DataAugumentation=args.p_augumentation,
            ave_data_flg=args.ave_data_flg,
            datalength=args.data_len,
        )

        # --- 訓練データをさらに訓練用と検証用に分割 ---
        if len(train_dataset) > 1:
            val_size = int(len(train_dataset) * 0.2)
            if val_size == 0:
                val_size = 1
            train_size = len(train_dataset) - val_size
            train_fold_dataset, val_fold_dataset = random_split(
                train_dataset, [train_size, val_size]
            )
        else:  # 訓練データが少ない場合は検証セットを作らない
            train_fold_dataset = train_dataset
            val_fold_dataset = train_dataset  # 検証も同じデータで行う

        # --- DataLoaderを作成 ---
        train_loader = DataLoader(
            train_fold_dataset, batch_size=args.batch_size, shuffle=True
        )
        val_loader = DataLoader(
            val_fold_dataset, batch_size=args.batch_size, shuffle=False
        )
        test_loader = DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False
        )

        # --- Foldごとにモデルとオプティマイザを初期化 ---
        model = MultiHeadVAE(input_dim=args.data_len, latent_dim=args.latent_dim).to(
            device
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = ReduceLROnPlateau(
            optimizer, "min", factor=0.1, patience=7, verbose=True
        )
        model_path = f"best_model_fold_{fold+1}_{target_name}.pt"
        early_stopping = EarlyStopping(
            patience=args.patience, verbose=True, path=model_path
        )

        # ==========================================================
        #                 エポックごとの学習ループ
        # ==========================================================
        for epoch in range(1, args.epochs + 1):
            train_loss, _, _ = train_multi_head_vae(
                model, train_loader, optimizer, device, beta=args.beta
            )
            val_loss, _, _ = evaluate_multi_head_vae(
                model, val_loader, device, beta=args.beta
            )
            print(
                f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
            )

            scheduler.step(val_loss)
            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                print("Early stopping triggered.")
                break

        # ==========================================================
        #           テスト (Foldごとの最終評価)
        # ==========================================================
        print(f"\nLoading best model from {model_path} for final test...")
        model.load_state_dict(torch.load(model_path, map_location=device))
        test_loss, _, _ = evaluate_multi_head_vae(
            model, test_loader, device, beta=args.beta
        )
        print(f"✅ Test Loss for Fold {fold+1} ({target_name}): {test_loss:.4f}")
        fold_test_losses.append(test_loss)

    # ==========================================================
    #                最終結果の集計
    # ==========================================================
    avg_test_loss = np.mean(fold_test_losses)
    std_test_loss = np.std(fold_test_losses)
    print(f"\n{'='*25} LOO Cross-Validation Finished {'='*25}")
    print(
        f"Average Test Loss across all {n_subjects} folds: {avg_test_loss:.4f} ± {std_test_loss:.4f}"
    )


if __name__ == "__main__":
    main()
