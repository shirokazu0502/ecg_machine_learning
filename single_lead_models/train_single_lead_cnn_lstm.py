import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import datetime
import json
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import os
import time

import utils
from models import CNN_LSTM_Model
import arguments as config
import Dataset

# Define the 8 target leads for the models
TARGET_LEADS = ["A1", "A2", "V1", "V2", "V3", "V4", "V5", "V6"]


def train_single_lead(
    model,
    train_loader,
    optimizer,
    scheduler,
    criterion,
    epochs,
    device,
    writer,
    target_lead,
):
    model.train()
    print(f"--- Training model for lead: {target_lead} ---")
    for epoch in range(epochs):
        total_train_loss = 0.0
        for x, xo, _, _ in train_loader:
            x, xo = x.to(device), xo.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, xo)
            loss.backward()
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        avg_loss = total_train_loss / len(train_loader)
        if (epoch + 1) % 10 == 0:
            print(
                f"Lead: {target_lead}, Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.6f}"
            )
        if writer:
            writer.add_scalar(f"Loss/train_{target_lead}", avg_loss, epoch)


def test_all_leads(test_loader_full, device, args, exp_dir):
    models = {}
    for lead in TARGET_LEADS:
        model = CNN_LSTM_Model(num_channels=args.num_channels, out_channels=1).to(
            device
        )
        model_path = os.path.join("model_pth", f"cnn_lstm_{lead}.pth")
        try:
            model.load_state_dict(
                torch.load(model_path, map_location=device, weights_only=True)
            )
        except FileNotFoundError:
            print(
                f"Error: Model for lead {lead} not found at {model_path}. Cannot perform test."
            )
            return

        model.eval()
        models[lead] = model

    (
        label_temps,
        test_val_all_mae,
        test_val_12ch_all_mae,
        test_val_all_rmse,
        test_val_12ch_all_rmse,
        pearson_scores,
        relative_roughness_scores,
    ) = ([], [], [], [], [], [], [])
    pearson_per_channel_scores = [[] for _ in range(len(TARGET_LEADS))]
    datalength = args.datalength
    ecg_ch_num = len(TARGET_LEADS)

    with torch.no_grad():
        for j, (x, xo_full, label_name, pt_index) in enumerate(test_loader_full):
            x, xo_full = x.to(device), xo_full.to(device)

            recon_x_leads = [models[lead](x) for lead in TARGET_LEADS]
            recon_x = torch.cat(recon_x_leads, dim=1)

            for i in range(len(label_name)):
                label_temps.append(label_name[i])

            mae_loss = utils.MAE_2(reduction="none")
            acc_mae = mae_loss(recon_x, xo_full)
            test_val_mae = utils.cul_val_no_pt(acc_mae)
            test_val_all_mae.extend([item.item() for item in test_val_mae])
            test_val_12ch_mae_np = np.array(
                [item.tolist() for item in utils.cul_val_per_12ch_no_pt(acc_mae)],
                dtype=np.float32,
            )
            test_val_12ch_all_mae.extend(test_val_12ch_mae_np.tolist())

            mse_loss = torch.nn.MSELoss(reduction="none")
            acc_mse = mse_loss(recon_x, xo_full)
            acc_rmse = torch.sqrt(acc_mse + 1e-8)
            test_val_rmse = utils.cul_val_no_pt(acc_rmse)
            test_val_all_rmse.extend([item.item() for item in test_val_rmse])
            test_val_12ch_rmse_np = np.array(
                [item.tolist() for item in utils.cul_val_per_12ch_no_pt(acc_mse)],
                dtype=np.float32,
            )
            test_val_12ch_all_rmse.extend(test_val_12ch_rmse_np.tolist())

            r, _ = utils.pearsonr(
                recon_x.flatten().cpu().numpy(), xo_full.flatten().cpu().numpy()
            )
            pearson_scores.append(r)

            for p_batch in range(recon_x.shape[0]):
                for q_ch in range(ecg_ch_num):
                    r_ch, _ = utils.pearsonr(
                        recon_x[p_batch, q_ch, :].cpu().numpy(),
                        xo_full[p_batch, q_ch, :].cpu().numpy(),
                    )
                    pearson_per_channel_scores[q_ch].append(r_ch)

            relative_roughness_scores.append(
                utils.calculate_relative_roughness(recon_x, xo_full)
            )

            utils.plot_fig_test_name_8ch_2row(
                recon_x=recon_x,
                xo=xo_full,
                datalength=datalength,
                exp_dir=exp_dir,
                args=args,
                batch_size_num=xo_full.shape[0],
                label_name=label_name,
                acc=test_val_12ch_mae_np,
                pt_index=pt_index,
                ecg_ch_names=TARGET_LEADS,
            )

    final_metrics = {
        "TARGET_NAME": args.TARGET_NAME,
        "MAE_all": np.mean(test_val_all_mae),
        "RMSE_all": np.mean(test_val_all_rmse),
        "pearson_all": np.mean(pearson_scores),
        "Relative_Roughness": np.mean(relative_roughness_scores),
    }
    for i, lead in enumerate(TARGET_LEADS):
        final_metrics[f"MAE_{lead}"] = np.mean(np.array(test_val_12ch_all_mae)[:, i])
        final_metrics[f"RMSE_{lead}"] = np.mean(np.array(test_val_12ch_all_rmse)[:, i])
        final_metrics[f"pearson_{lead}"] = np.mean(pearson_per_channel_scores[i])

    output_file = os.path.join(
        args.mae_folder,
        f"MAE_single_lead_ensemble_{args.Dataset_name}_{args.TARGET_NAME}.csv",
    )
    utils.write_to_csv(output_file, data=final_metrics)

    print("\n--- METRICS ---")
    for key, value in sorted(final_metrics.items()):
        print(f"  {key}: {value}")
    print("--- END METRICS ---")


def main():
    args = config.get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    model_save_dir = "model_pth"
    utils.create_directory_if_not_exists(model_save_dir)

    all_subjects = utils.get_all_subject_names(args.Dataset_name)
    if args.TARGET_NAME not in all_subjects:
        raise ValueError(f"TARGET_NAME '{args.TARGET_NAME}' not found in the dataset.")

    training_subjects = [s for s in all_subjects if s != args.TARGET_NAME]
    test_subjects = [args.TARGET_NAME]

    if args.mode == "train":
        print(
            f"\n--- STARTING TRAINING FOR ALL 8 LEADS (Test Subject: {args.TARGET_NAME}) ---"
        )
        for target_lead_name in TARGET_LEADS:
            writer = SummaryWriter(
                log_dir=f"runs/{args.current_time}/{args.TARGET_NAME}/CNN_LSTM_{target_lead_name}"
            )

            common_dataset_args = {
                "Dataset_name": args.Dataset_name,
                "dataset_num": args.dataset_num,
                "DataAugmentation": args.DataAugmentation,
                "ave_data_flg": args.ave_data_flg,
                "num_channels": args.num_channels,
            }

            train_dataset = Dataset.Dataset_setup_8ch_pt_augmentation(
                include_subjects=training_subjects,
                target_lead=target_lead_name,
                **common_dataset_args,
            )

            train_loader = DataLoader(
                dataset=train_dataset, batch_size=args.train_batch_size, shuffle=True
            )
            model = CNN_LSTM_Model(num_channels=args.num_channels, out_channels=1).to(
                device
            )
            optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
            criterion = utils.loss_fn_lstm

            train_single_lead(
                model,
                train_loader,
                optimizer,
                scheduler,
                criterion,
                args.epochs,
                device,
                writer,
                target_lead_name,
            )

            save_path = os.path.join(model_save_dir, f"cnn_lstm_{target_lead_name}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Saved model for lead {target_lead_name} to {save_path}")
            writer.close()

        print("\n--- FINISHED TRAINING FOR ALL 8 LEADS ---")

    elif args.mode == "test":
        print(f"\n--- STARTING TESTING for test subject: {args.TARGET_NAME} ---")

        ts_test = f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_test_{args.TARGET_NAME}"
        exp_dir_test = os.path.join(args.fig_root, str(ts_test))
        utils.create_directory_if_not_exists(exp_dir_test)

        test_dataset_full = Dataset.Dataset_setup_8ch_pt_augmentation(
            include_subjects=test_subjects,
            Dataset_name=args.Dataset_name,
            dataset_num=args.dataset_num,
            DataAugmentation="",
            ave_data_flg=args.ave_data_flg,
            num_channels=args.num_channels,
            target_lead=None,
        )
        test_loader_full = DataLoader(test_dataset_full, batch_size=4, shuffle=False)

        test_all_leads(test_loader_full, device, args, exp_dir_test)

        print(f"\n--- FINISHED TESTING for test subject: {args.TARGET_NAME} ---")


if __name__ == "__main__":
    main()
