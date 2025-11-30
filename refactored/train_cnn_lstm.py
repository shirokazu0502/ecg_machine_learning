import torch
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from collections import defaultdict
import datetime
import json
import numpy as np
from torch.utils.tensorboard import SummaryWriter

import utils
from models import CNN_LSTM_Model  # UNet1DからCNN_LSTM_Modelに変更
import arguments as config
import Dataset
import os
import time


def train_cnn_lstm(  # 関数名を変更
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    criterion,
    epochs,
    device,
    writer,
    early_stopping,
):
    model.train()
    save_path = os.path.join("model_pth", "cnn_lstm.pth")  # 保存パスを変更
    # early_stopping = EarlyStopping(patience=150, verbose=True, path=save_path)
    for epoch in range(epochs):
        total_train_loss = 0.0
        model.train()
        for x, xo, _, _ in train_loader:
            x, xo = x.to(device), xo.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, xo)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        scheduler.step()

        # # --- Validation Loop for Early Stopping ---
        # model.eval()
        # total_val_loss = 0.0
        # with torch.no_grad():
        #     for x, xo, _, _ in val_loader:
        #         x, xo = x.to(device), xo.to(device)
        #         outputs = model(x)
        #         loss = criterion(outputs, xo)
        #         total_val_loss += loss.item()
        #
        # avg_val_loss = total_val_loss / len(val_loader)
        # writer.add_scalar("Loss/val_loss", avg_val_loss, epoch)
        #
        # early_stopping(avg_val_loss, model)
        # if early_stopping.early_stop:
        #     print("Early stopping")
        #     break
        # # --- End Validation Loop ---

        print(
            f"Epoch [{epoch + 1}/{epochs}], Loss: {total_train_loss / len(train_loader):.4f}"
        )
        writer.add_scalar(
            "Loss/train_loss", total_train_loss / len(train_loader), epoch
        )


def test_cnn_lstm(model, test_loader, device, args, exp_dir):  # 関数名を変更
    model.eval()
    all_outputs = []
    all_targets = []
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
    if ecg_ch == 12:
        ecg_ch_names = [
            "A1",
            "A2",
            "V1",
            "V2",
            "V3",
            "V4",
            "V5",
            "V6",
        ]
    if ecg_ch == 8:
        ecg_ch_names = ["A1", "A2", "V1", "V2", "V3", "V4", "V5", "V6"]
    with torch.no_grad():
        for j, (x, xo, label_name, pt_index) in enumerate(test_loader):
            x, xo = x.to(device), xo.to(device)
            recon_x = model(x)
            for i in range(len(label_name)):
                label_temps.append(label_name[i])
            numplotfig = len(x)
            mae_loss = utils.MAE_2(reduction="none")
            recon_x = recon_x.reshape(-1, ecg_ch, datalength)
            xo = xo.reshape(-1, ecg_ch, datalength)
            acc_mae = mae_loss(recon_x, xo)

            # args.loss_pt_on_off は run_unet.sh から削除されたため、ここでは考慮しない
            test_val_mae = utils.cul_val_no_pt(acc_mae)

            test_val_mae = [array_item.item() for array_item in test_val_mae]
            test_val_mae = np.array(test_val_mae, dtype=np.float32)
            test_val_all_mae += test_val_mae.tolist()

            # args.loss_pt_on_off は run_unet.sh から削除されたため、ここでは考慮しない
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

            # args.loss_pt_on_off は run_unet.sh から削除されたため、ここでは考慮しない
            test_val_rmse = utils.cul_val_no_pt(acc_rmse)

            test_val_rmse = [array_item.item() for array_item in test_val_rmse]
            test_val_rmse = np.array(test_val_rmse, dtype=np.float32)
            test_val_all_rmse += test_val_rmse.tolist()

            # args.loss_pt_on_off は run_unet.sh から削除されたため、ここでは考慮しない
            test_val_12ch_rmse = utils.cul_val_per_12ch_no_pt(acc_rmse)

            test_val_12ch_rmse = np.array(
                [array_item.tolist() for array_item in test_val_12ch_rmse],
                dtype=np.float32,
            )
            test_val_12ch_rmse = test_val_12ch_rmse.reshape(-1, args.ecg_ch_num)
            test_val_12ch_all_rmse += test_val_12ch_rmse.tolist()

            recon_x_np_flat = (
                recon_x.reshape(-1, datalength).cpu().numpy().astype(np.float64)
            )
            xo_np_flat = xo.reshape(-1, datalength).cpu().numpy().astype(np.float64)

            r, _ = utils.pearsonr(
                recon_x_np_flat.ravel(),
                xo_np_flat.ravel(),
            )
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

            batch_size_now = xo.shape[0]

            utils.plot_fig(
                numplotfig=batch_size_now,
                recon_x=recon_x,
                xo=xo,
                datalength=datalength,
                exp_dir=exp_dir,
                args=args,
                label_name=label_name,
                ecg_ch_names=ecg_ch_names,
            )
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

    return (
        test_val_all_mae,
        test_val_12ch_all_mae,
        test_val_all_rmse,
        test_val_12ch_all_rmse,
        pearson_scores,
        relative_roughness_scores,
        pearson_per_channel_scores,
    )


def main():
    utils.create_directory_if_not_exists("model_pth")
    args = config.get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # current_time文字列をCNN_LSTM用に変更
    args.current_time = datetime.date.today().strftime(
        "CNN_LSTM_%Y%m%d%H_15ch_arrange_direction"
    )
    writer = SummaryWriter(log_dir=f"runs/{args.current_time}/{args.TARGET_NAME}")

    train_dataset, test_dataset = Dataset.Dataset_setup_8ch_pt_augmentation(
        TARGET_NAME=args.TARGET_NAME,
        Dataset_name=args.Dataset_name,
        dataset_num=args.dataset_num,
        DataAugmentation=args.DataAugmentation,
        ave_data_flg=args.ave_data_flg,
        num_channels=args.num_channels,
    )

    # ts文字列をCNN_LSTM用に変更し、不要な引数を削除
    ts = (
        datetime.datetime.fromtimestamp(time.time()).strftime("%Y%m%d_%H%M%S")
        + "_Dataset_name="
        + args.Dataset_name
        + "_"
        + args.mode
        + "_TARGETNAME="
        + str(args.TARGET_NAME)
        + "_model=CNN_LSTM"  # モデル名を明示的に追加
    )

    exp_dir = os.path.join(args.fig_root, str(ts))
    utils.create_directory_if_not_exists(exp_dir)

    with open(os.path.join(exp_dir, "args.json"), mode="w") as f:
        json.dump(args.__dict__, f, indent=4)

    out_channels = 8
    # UNet1Dのbase_filtersは不要。CNN_LSTM_Modelの引数を渡す
    common_kwargs = {
        "num_channels": args.num_channels,
        "out_channels": out_channels,
        "cnn_filters_1": args.cnn_filters_1,
        "cnn_filters_2": args.cnn_filters_2,
        "cnn_kernel_size": args.cnn_kernel_size,
        "lstm_hidden_size": args.lstm_hidden_size,
        "lstm_num_layers": args.lstm_num_layers,
    }
    model = CNN_LSTM_Model(**common_kwargs).to(device)  # UNet1DからCNN_LSTM_Modelに変更

    if args.mode == "train":
        print("TRAINING MODE::\n")

        # # --- Create Validation Set ---
        # val_percent = 0.1
        # n_val = int(len(train_dataset) * val_percent)
        # n_train = len(train_dataset) - n_val
        # train_subset, val_subset = random_split(train_dataset, [n_train, n_val])
        #
        # train_loader = DataLoader(
        #     dataset=train_subset, batch_size=args.train_batch_size, shuffle=True
        # )
        # val_loader = DataLoader(val_subset, args.val_batch_size, shuffle=False)
        # # --- End Create Validation Set ---

        train_loader = DataLoader(
            dataset=train_dataset, batch_size=args.train_batch_size, shuffle=True
        )
        test_loader = DataLoader(test_dataset, args.val_batch_size, shuffle=False)

        # # --- Instantiate EarlyStopping ---
        # save_path = os.path.join("model_pth", "unet.pth")
        # early_stopping = utils.EarlyStopping(patience=20, verbose=True, path=save_path)
        # # --- End Instantiate EarlyStopping ---

        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.learning_rate
        )  # unetからmodelに変更
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)

        print("Training CNN_LSTM...")  # ログメッセージを変更
        train_cnn_lstm(  # 関数名を変更
            model,
            train_loader,
            test_loader,  # Replace with val_loader when enabling early stopping
            optimizer,
            scheduler,
            utils.loss_fn_lstm,  # loss_fn_unetからloss_fn_lstmに変更
            args.epochs,
            device,
            writer,
            None,  # Replace with early_stopping when enabling
        )

        torch.save(
            model.state_dict(),  # unetからmodelに変更
            os.path.join("model_pth", "cnn_lstm.pth"),  # 保存パスを変更
        )
        torch.save(
            model.state_dict(),  # unetからmodelに変更
            os.path.join(args.fig_root, str(ts), "cnn_lstm.pth"),  # 保存パスを変更
        )

        writer.close()

    elif args.mode == "test":
        print("TEST MODE::\n")
        test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
        try:
            model.load_state_dict(
                torch.load(
                    os.path.join("model_pth", "cnn_lstm.pth"),  # 保存パスを変更
                    map_location=lambda storage, loc: storage,
                )
            )
            print(
                "Loaded pre-trained model from model_pth/cnn_lstm.pth"
            )  # ログメッセージを変更
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
        ) = test_cnn_lstm(
            model, test_loader, device, args, exp_dir
        )  # 関数名を変更

        test_val_all_mae = np.array(test_val_all_mae)
        test_val_mean_mae = np.mean(test_val_all_mae)
        test_val_12ch_all_mae = np.array(test_val_12ch_all_mae)
        test_val_12ch_mean_mae = np.mean(test_val_12ch_all_mae, axis=0)
        pearson_all_score = np.mean(pearson_scores)
        mean_relative_roughness = np.mean(relative_roughness_scores)
        mean_pearson_per_channel = [
            np.mean(scores) if scores else np.nan
            for scores in pearson_per_channel_scores
        ]

        test_val_all_rmse = np.array(test_val_all_rmse)
        test_val_rmse_mean = np.mean(test_val_all_rmse)
        test_val_12ch_all_rmse = np.array(test_val_12ch_all_rmse)
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
            args.mae_folder
            + "/MAE_leave_1_out_{}_PRTweight_{}_{}_{}_augumentation={}.csv".format(
                args.Dataset_name,
                str(
                    args.loss_pt_on_off_P_weight
                ),  # loss_pt_on_off_P_weightはarguments.pyに残っているためそのまま
                str(
                    args.loss_pt_on_off_R_weight
                ),  # loss_pt_on_off_R_weightはarguments.pyに残っているためそのまま
                str(
                    args.loss_pt_on_off_T_weight
                ),  # loss_pt_on_off_T_weightはarguments.pyに残っているためそのまま
                args.p_augumentation,
                args.r_augumentation,
                args.t_augumentation,
            )
        )
        utils.write_to_csv(output_file, data=data_to_write)

        print("--- METRICS ---")
        with open(output_file, "r") as f:
            print(f.read())
        print("--- END METRICS ---")


if __name__ == "__main__":
    main()
