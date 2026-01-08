import sys
import os
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import datetime
import json
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import time
from torch.cuda.amp import autocast, GradScaler

# Add the project root to the Python path
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

import utils
from models import ECGReconGNN  # Changed from SimpleCNN
import arguments as config
import Dataset

# --- 学習ループ外での初期化 ---
use_amp = torch.cuda.is_available()
scaler = GradScaler(enabled=use_amp)


def train_gnn(
    model,
    train_loader,
    optimizer,
    scheduler,
    criterion,
    epochs,
    device,
    writer,
):
    model.train()
    for epoch in range(epochs):
        total_train_loss = 0.0
        model.train()
        for batch_idx, (x_input, x_output, _, _) in enumerate(train_loader):
            x_input, x_output = x_input.to(device, non_blocking=True), x_output.to(
                device, non_blocking=True
            )

            # Input NaN/Inf Check
            if torch.isnan(x_input).any() or torch.isinf(x_input).any():
                print(
                    f"!!! Warning: NaN/Inf in training input at Epoch {epoch}, Batch {batch_idx}. Skipping."
                )
                continue

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=use_amp):
                outputs = model(x_input)
                loss = criterion(outputs, x_output)

            if torch.isnan(loss) or torch.isinf(loss):
                print(
                    f"!!! Warning: NaN/Inf loss detected at Epoch {epoch}, Batch {batch_idx}. Skipping batch."
                )
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            total_train_loss += loss.item()

        scheduler.step()

        avg_loss = total_train_loss / len(train_loader) if len(train_loader) > 0 else 0
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.6f}")
        writer.add_scalar("Loss/train_loss", avg_loss, epoch)


def test_gnn(model, test_loader, device, args, exp_dir):
    model.eval()
    label_temps = []
    test_val_all_mae = []
    test_val_12ch_all_mae = []
    test_val_all_rmse = []
    test_val_12ch_all_rmse = []
    pearson_scores = []
    relative_roughness_scores = []
    pearson_per_channel_scores = [[] for _ in range(args.ecg_ch_num)]
    datalength = args.datalength
    ecg_ch = args.ecg_ch_num
    ecg_ch_names = ["A1", "A2", "V1", "V2", "V3", "V4", "V5", "V6"]

    with torch.no_grad():
        for j, (x, xo, label_name, pt_index) in enumerate(test_loader):
            x, xo = x.to(device), xo.to(device)

            with autocast(enabled=use_amp):
                recon_x = model(x)

            # Skip batch if model output is NaN/Inf
            if torch.isnan(recon_x).any() or torch.isinf(recon_x).any():
                print(
                    f"!!! Warning: NaN/Inf detected in model output during test for labels {label_name}. Skipping batch."
                )
                continue

            for i in range(len(label_name)):
                label_temps.append(label_name[i])

            mae_loss = utils.MAE_2(reduction="none")
            recon_x = recon_x.reshape(-1, ecg_ch, datalength)
            xo = xo.reshape(-1, ecg_ch, datalength)
            acc_mae = mae_loss(recon_x, xo)

            test_val_mae = utils.cul_val_no_pt(acc_mae)
            test_val_all_mae.extend([item.item() for item in test_val_mae])

            test_val_12ch_mae = utils.cul_val_per_12ch_no_pt(acc_mae)
            test_val_12ch_all_mae.extend([item.tolist() for item in test_val_12ch_mae])

            mse_loss = torch.nn.MSELoss(reduction="none")
            acc_mse = mse_loss(recon_x, xo)
            acc_rmse = torch.sqrt(acc_mse + 1e-8)

            test_val_rmse = utils.cul_val_no_pt(acc_rmse)
            test_val_all_rmse.extend([item.item() for item in test_val_rmse])

            test_val_12ch_rmse = utils.cul_val_per_12ch_no_pt(acc_rmse)
            test_val_12ch_all_rmse.extend(
                [item.tolist() for item in test_val_12ch_rmse]
            )

            recon_x_np_flat = (
                recon_x.reshape(-1, datalength).cpu().numpy().astype(np.float64)
            )
            xo_np_flat = xo.reshape(-1, datalength).cpu().numpy().astype(np.float64)

            r, _ = utils.pearsonr(recon_x_np_flat.ravel(), xo_np_flat.ravel())
            pearson_scores.append(r)

            recon_x_np_ch = recon_x.cpu().numpy().astype(np.float64)
            xo_np_ch = xo.cpu().numpy().astype(np.float64)
            for p_batch in range(recon_x_np_ch.shape[0]):
                for q_ch in range(ecg_ch):
                    r_ch, _ = utils.pearsonr(
                        recon_x_np_ch[p_batch, q_ch, :], xo_np_ch[p_batch, q_ch, :]
                    )
                    pearson_per_channel_scores[q_ch].append(r_ch)

            relative_roughness = utils.calculate_relative_roughness(recon_x, xo)
            relative_roughness_scores.append(relative_roughness)

            utils.plot_fig_test_name_8ch_2row(
                recon_x=recon_x,
                xo=xo,
                datalength=datalength,
                exp_dir=exp_dir,
                args=args,
                batch_size_num=xo.shape[0],
                label_name=label_name,
                acc=acc_mae.mean(dim=(1, 2)),
                pt_index=pt_index,
                ecg_ch_names=ecg_ch_names,
            )
            utils.save_csv2(
                data=recon_x,
                args=args,
                exp_dir=exp_dir,
                label_name=label_name,
                data_rec_or_xo="recon_x",
            )
            utils.save_csv2(
                data=xo,
                args=args,
                exp_dir=exp_dir,
                label_name=label_name,
                data_rec_or_xo="xo",
            )

    # --- Aggregate and Save Metrics ---
    test_val_mean_mae = np.mean(test_val_all_mae) if test_val_all_mae else np.nan
    test_val_12ch_mean_mae = (
        np.mean(test_val_12ch_all_mae, axis=0)
        if test_val_12ch_all_mae
        else [np.nan] * ecg_ch
    )
    pearson_all_score = np.mean(pearson_scores) if pearson_scores else np.nan
    mean_relative_roughness = (
        np.mean(relative_roughness_scores) if relative_roughness_scores else np.nan
    )
    mean_pearson_per_channel = [
        np.mean(scores) if scores else np.nan for scores in pearson_per_channel_scores
    ]
    test_val_rmse_mean = np.mean(test_val_all_rmse) if test_val_all_rmse else np.nan
    test_val_12ch_rmse_mean = (
        np.mean(test_val_12ch_all_rmse, axis=0)
        if test_val_12ch_all_rmse
        else [np.nan] * ecg_ch
    )

    data_to_write = {
        "TARGET_NAME": args.TARGET_NAME,
        "MAE_all": test_val_mean_mae,
        "RMSE_all": test_val_rmse_mean,
        "pearson_all": pearson_all_score,
        "Relative_Roughness": mean_relative_roughness,
    }
    for i, name in enumerate(ecg_ch_names):
        data_to_write[f"MAE_{name}"] = (
            test_val_12ch_mean_mae[i] if len(test_val_12ch_mean_mae) > i else np.nan
        )
        data_to_write[f"RMSE_{name}"] = (
            test_val_12ch_rmse_mean[i] if len(test_val_12ch_rmse_mean) > i else np.nan
        )
        data_to_write[f"pearson_{name}"] = (
            mean_pearson_per_channel[i] if len(mean_pearson_per_channel) > i else np.nan
        )

    output_file = os.path.join(
        args.mae_folder, f"MAE_leave_1_out_{args.Dataset_name}_GNN.csv"
    )  # Changed name
    utils.write_to_csv(output_file, data=data_to_write)

    print("--- METRICS ---")
    print(data_to_write)
    print("--- END METRICS ---")


def main():
    utils.create_directory_if_not_exists("model_pth")
    args = config.get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    writer = SummaryWriter(
        f"runs/{args.current_time}/{args.TARGET_NAME}/GNN_log"
    )  # Changed name

    train_dataset, test_dataset = Dataset.Dataset_setup_8ch_pt_augmentation(
        TARGET_NAME=args.TARGET_NAME,
        Dataset_name=args.Dataset_name,
        dataset_num=args.dataset_num,
        DataAugmentation=args.DataAugmentation,
        ave_data_flg=args.ave_data_flg,
        num_channels=args.num_channels,
    )

    ts = (
        datetime.datetime.fromtimestamp(time.time()).strftime("%Y%m%d_%H%M%S")
        + f"_Dataset_name={args.Dataset_name}"
        + f"_transform_type={args.transform_type}"
        + f"_{args.mode}"
        + f"_TARGETNAME={args.TARGET_NAME}"
        + f"_rnn_hidden={args.rnn_hidden_size}"
        + f"_gnn_hidden={args.gnn_hidden_size}"
    )

    exp_dir = os.path.join(args.fig_root, str(ts))
    utils.create_directory_if_not_exists(exp_dir)

    with open(os.path.join(exp_dir, "args.json"), mode="w") as f:
        json.dump(args.__dict__, f, indent=4)

    model = ECGReconGNN(
        input_dim=args.num_channels,
        hidden_dim_rnn=args.rnn_hidden_size,
        hidden_dim_gcn=args.gnn_hidden_size,
        output_dim=args.ecg_ch_num,
        sequence_length=args.datalength,
    ).to(device)

    model_path = os.path.join("model_pth", "gnn_model.pth")  # Changed name

    if args.mode == "train":
        print("TRAINING MODE:")
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=args.train_batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
        criterion = utils.loss_fn_mse_and_corr

        print("Training GNN Model...")
        train_gnn(
            model,
            train_loader,
            optimizer,
            scheduler,
            criterion,
            args.epochs,
            device,
            writer,
        )

        torch.save(model.state_dict(), model_path)
        torch.save(model.state_dict(), os.path.join(exp_dir, "gnn_model.pth"))
        writer.close()

    elif args.mode == "test":
        print("TEST MODE:")
        test_loader = DataLoader(
            test_dataset, batch_size=args.test_batch_size, shuffle=False
        )
        try:
            model.load_state_dict(
                torch.load(model_path, map_location=lambda storage, loc: storage)
            )
            print(f"Loaded pre-trained model from {model_path}")
        except (FileNotFoundError, RuntimeError) as e:
            print(
                f"Could not load pre-trained model: {e}. Continuing with initial model."
            )
        test_gnn(model, test_loader, device, args, exp_dir)


if __name__ == "__main__":
    main()
