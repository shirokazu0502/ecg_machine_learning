import sys
import os

# Add the project root to the Python path
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from collections import defaultdict
import datetime
import json
import numpy as np
from torch.utils.tensorboard import SummaryWriter

import utils
from models import VAE
import arguments as config
import Dataset
import time


def train_simple_vae(
    model,
    train_loader,
    optimizer,
    criterion,
    epochs,
    device,
    writer,
    args,
):
    model.train()
    for epoch in range(epochs):
        total_train_loss = 0.0
        for x, xo, _, _ in train_loader:
            x, xo = x.to(device), xo.to(device)
            optimizer.zero_grad()
            recon_x, mean, log_var, z = model(x)
            loss, mse, kdl = criterion(
                recon_x, xo, mean, log_var, args.datalength, args
            )
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_train_loss:.4f}")
        writer.add_scalar("Loss/train", avg_train_loss, epoch)


def test_simple_vae(model, test_loader, device, args, exp_dir):
    model.eval()
    label_temps = []
    test_val_all_mae = []
    test_val_12ch_all_mae = []
    test_val_all_rmse = []
    test_val_12ch_all_rmse = []
    batch_pearson_scores = []
    all_pearson_scores = []
    relative_roughness_scores = []
    pearson_per_channel_scores = [[] for _ in range(args.ecg_ch_num)]
    datalength = args.datalength
    ecg_ch = args.ecg_ch_num
    ecg_ch_names = ["A1", "A2", "V1", "V2", "V3", "V4", "V5", "V6"]

    with torch.no_grad():
        for j, (x, xo, label_name, pt_index) in enumerate(test_loader):
            x, xo = x.to(device), xo.to(device)
            recon_x, means, log_var, z = model(x)
            for i in range(len(label_name)):
                label_temps.append(label_name[i])
            numplotfig = len(x)
            mae_loss = utils.MAE_2(reduction="none")
            recon_x = recon_x.reshape(-1, ecg_ch, datalength)
            xo = xo.reshape(-1, ecg_ch, datalength)
            acc_mae = mae_loss(recon_x, xo)

            test_val_mae = utils.cul_val_no_pt(acc_mae)
            test_val_mae = [array_item.item() for array_item in test_val_mae]
            test_val_mae = np.array(test_val_mae, dtype=np.float32)
            test_val_all_mae += test_val_mae.tolist()

            test_val_12ch_mae = utils.cul_val_per_12ch_no_pt(acc_mae)
            test_val_12ch_mae = np.array(
                [array_item.tolist() for array_item in test_val_12ch_mae],
                dtype=np.float32,
            )
            test_val_12ch_mae = test_val_12ch_mae.reshape(-1, args.ecg_ch_num)
            test_val_12ch_all_mae += test_val_12ch_mae.tolist()

            mse_loss = torch.nn.MSELoss(reduction="none")
            acc_mse = mse_loss(recon_x, xo)
            acc_rmse = torch.sqrt(acc_mse + 1e-8)

            test_val_rmse = utils.cul_val_no_pt(acc_rmse)
            test_val_rmse = [array_item.item() for array_item in test_val_rmse]
            test_val_rmse = np.array(test_val_rmse, dtype=np.float32)
            test_val_all_rmse += test_val_rmse.tolist()

            test_val_12ch_rmse = utils.cul_val_per_12ch_no_pt(acc_rmse)
            test_val_12ch_rmse = np.array(
                [array_item.tolist() for array_item in test_val_12ch_rmse],
                dtype=np.float32,
            )
            test_val_12ch_rmse = test_val_12ch_rmse.reshape(-1, args.ecg_ch_num)
            test_val_12ch_all_rmse += test_val_12ch_rmse.tolist()

            recon_x_np_ch = recon_x.cpu().numpy().astype(np.float64)
            xo_np_ch = xo.cpu().numpy().astype(np.float64)
            for p_batch in range(recon_x_np_ch.shape[0]):
                for q_ch in range(ecg_ch):
                    r_ch, _ = utils.pearsonr(
                        recon_x_np_ch[p_batch, q_ch, :], xo_np_ch[p_batch, q_ch, :]
                    )
                    pearson_per_channel_scores[q_ch].append(r_ch)

                    # 各チャネルの相関係数を追加
                    batch_pearson_scores.append(r_ch)

            # バッチ内の平均相関係数を計算して追加
            mean_batch_pearson = np.mean(batch_pearson_scores)
            all_pearson_scores.append(mean_batch_pearson)

            relative_roughness = utils.calculate_relative_roughness(recon_x, xo)
            relative_roughness_scores.append(relative_roughness)

            batch_size_now = xo.shape[0]

            utils.plot_fig_test_name_8ch_2row(
                recon_x=recon_x,
                xo=xo,
                datalength=datalength,
                exp_dir=exp_dir,
                args=args,
                batch_size_num=batch_size_now,
                label_name=label_name,
                acc=test_val_12ch_mae,
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
        test_val_all_mae = np.array(test_val_all_mae)
        test_val_12ch_all_mae = np.array(test_val_12ch_all_mae)

    return (
        test_val_all_mae,
        test_val_12ch_all_mae,
        test_val_all_rmse,
        test_val_12ch_all_rmse,
        all_pearson_scores,
        relative_roughness_scores,
        pearson_per_channel_scores,
    )


def main():
    utils.create_directory_if_not_exists("model_pth")
    args = config.get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    print(args.num_channels)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    writer = SummaryWriter(
        log_dir=f"runs/{args.current_time}/{args.TARGET_NAME}/SimpleVAE_log"
    )

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
        + "_SimpleVAE"
        + "_TARGETNAME="
        + str(args.TARGET_NAME)
    )

    exp_dir = os.path.join(args.fig_root, str(ts))
    utils.create_directory_if_not_exists(exp_dir)

    with open(os.path.join(exp_dir, "args.json"), mode="w") as f:
        json.dump(args.__dict__, f, indent=4)

    common_kwargs = {
        "datalength": args.datalength,
        "enc_convlayer_sizes": args.enc_convlayer_sizes,
        "enc_fclayer_sizes": args.enc_fclayer_sizes,
        "dec_fclayer_sizes": args.dec_fclayer_sizes,
        "dec_convlayer_sizes": args.dec_convlayer_sizes,
        "latent_size": args.latent_size,
        "conditional": args.conditional,
        "num_labels": 20 if args.conditional else 0,
        "num_channels": args.num_channels,
    }

    model = VAE(**common_kwargs).to(device)

    if args.mode == "train":
        print("TRAINING MODE::\n")
        train_loader = DataLoader(
            dataset=train_dataset, batch_size=args.train_batch_size, shuffle=True
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

        print("Training Simple VAE...")
        train_simple_vae(
            model,
            train_loader,
            optimizer,
            utils.loss_fn_mse_kld,  # Using standard MSE loss for simplicity
            args.epochs,
            device,
            writer,
            args,
        )

        torch.save(
            model.state_dict(),
            os.path.join("model_pth", "vae_simple.pth"),
        )
        writer.close()

    elif args.mode == "test":
        print("TEST MODE::\n")
        test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
        try:
            model.load_state_dict(
                torch.load(
                    os.path.join("model_pth", "vae_simple.pth"),
                    map_location=lambda storage, loc: storage,
                )
            )
            print("Loaded pre-trained model from model_pth/vae_simple.pth")
        except (FileNotFoundError, RuntimeError) as e:
            print(
                f"Could not load pre-trained model: {e}. Continuing with initial model."
            )

        (
            test_val_all_mae,
            test_val_12ch_all_mae,
            test_val_all_rmse,
            test_val_12ch_all_rmse,
            pearson_scores,
            relative_roughness_scores,
            pearson_per_channel_scores,
        ) = test_simple_vae(model, test_loader, device, args, exp_dir)

        test_val_mean_mae = np.mean(test_val_all_mae)
        test_val_12ch_mean_mae = np.mean(test_val_12ch_all_mae, axis=0)
        pearson_all_score = np.mean(pearson_scores)
        mean_relative_roughness = np.mean(relative_roughness_scores)
        mean_pearson_per_channel = [
            np.mean(scores) if scores else np.nan
            for scores in pearson_per_channel_scores
        ]

        test_val_rmse_mean = np.mean(test_val_all_rmse)
        test_val_12ch_rmse_mean = np.mean(test_val_12ch_all_rmse, axis=0)

        data_to_write = {
            "TARGET_NAME": args.TARGET_NAME,
            "MAE_all": test_val_mean_mae,
            "MAE_A1": test_val_12ch_mean_mae[0],
            "MAE_A2": test_val_12ch_mean_mae[1],
            "MAE_V1": test_val_12ch_mean_mae[2],
            "MAE_V2": test_val_12ch_mean_mae[3],
            "MAE_V3": test_val_12ch_mean_mae[4],
            "MAE_V4": test_val_12ch_mean_mae[5],
            "MAE_V5": test_val_12ch_mean_mae[6],
            "MAE_V6": test_val_12ch_mean_mae[7],
            "RMSE_all": test_val_rmse_mean,
            "RMSE_A1": test_val_12ch_rmse_mean[0],
            "RMSE_A2": test_val_12ch_rmse_mean[1],
            "RMSE_V1": test_val_12ch_rmse_mean[2],
            "RMSE_V2": test_val_12ch_rmse_mean[3],
            "RMSE_V3": test_val_12ch_rmse_mean[4],
            "RMSE_V4": test_val_12ch_rmse_mean[5],
            "RMSE_V5": test_val_12ch_rmse_mean[6],
            "RMSE_V6": test_val_12ch_rmse_mean[7],
            "pearson_all": pearson_all_score,
            "pearson_A1": mean_pearson_per_channel[0],
            "pearson_A2": mean_pearson_per_channel[1],
            "pearson_V1": mean_pearson_per_channel[2],
            "pearson_V2": mean_pearson_per_channel[3],
            "pearson_V3": mean_pearson_per_channel[4],
            "pearson_V4": mean_pearson_per_channel[5],
            "pearson_V5": mean_pearson_per_channel[6],
            "pearson_V6": mean_pearson_per_channel[7],
            "Relative_Roughness": mean_relative_roughness,
        }
        output_file = os.path.join(
            args.mae_folder,
            f"MAE_SimpleVAE_{args.Dataset_name}.csv",
        )
        utils.write_to_csv(output_file, data=data_to_write)

        print("--- METRICS ---")
        with open(output_file, "r") as f:
            print(f.read())
        print("--- END METRICS ---")


if __name__ == "__main__":
    main()
