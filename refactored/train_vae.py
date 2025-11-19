import sys
import os

# Add the project root to the Python path
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

import torch
from torch.utils.data import DataLoader
from collections import defaultdict
import datetime
import json
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import time
import utils
from models import VAE
import arguments as config
import Dataset


def weighted_average_join(
    recon_x_p,
    recon_x_r,
    recon_x_t,
    p_end_orig=130,
    r_start_orig=130,
    r_end_orig=170,
    t_start_orig=170,
    overlap_pr=10,
    overlap_rt=10,
):
    final_recon_x = torch.zeros_like(recon_x_p)
    p_dominant_end = p_end_orig - overlap_pr // 2
    final_recon_x[:, :, :p_dominant_end] = recon_x_p[:, :, :p_dominant_end]

    pr_overlap_start = p_dominant_end
    pr_overlap_end = r_start_orig + overlap_pr // 2

    weights_p_fade_out = torch.linspace(
        1, 0, steps=overlap_pr, device=recon_x_p.device
    ).view(1, 1, -1)
    weights_r_fade_in = torch.linspace(
        0, 1, steps=overlap_pr, device=recon_x_p.device
    ).view(1, 1, -1)

    p_segment_for_pr_overlap = recon_x_p[
        :, :, pr_overlap_start : pr_overlap_start + overlap_pr
    ]
    r_segment_for_pr_overlap = recon_x_r[
        :, :, pr_overlap_start : pr_overlap_start + overlap_pr
    ]
    final_recon_x[:, :, pr_overlap_start : pr_overlap_start + overlap_pr] = (
        p_segment_for_pr_overlap * weights_p_fade_out
        + r_segment_for_pr_overlap * weights_r_fade_in
    )

    r_dominant_start = pr_overlap_start + overlap_pr
    r_dominant_end = r_end_orig - overlap_rt // 2
    final_recon_x[:, :, r_dominant_start:r_dominant_end] = recon_x_r[
        :, :, r_dominant_start:r_dominant_end
    ]

    rt_overlap_start = r_dominant_end

    weights_r_fade_out = torch.linspace(
        1, 0, steps=overlap_rt, device=recon_x_r.device
    ).view(1, 1, -1)
    weights_t_fade_in = torch.linspace(
        0, 1, steps=overlap_rt, device=recon_x_t.device
    ).view(1, 1, -1)

    r_segment_for_rt_overlap = recon_x_r[
        :, :, rt_overlap_start : rt_overlap_start + overlap_rt
    ]
    t_segment_for_rt_overlap = recon_x_t[
        :, :, rt_overlap_start : rt_overlap_start + overlap_rt
    ]

    final_recon_x[:, :, rt_overlap_start : rt_overlap_start + overlap_rt] = (
        r_segment_for_rt_overlap * weights_r_fade_out
        + t_segment_for_rt_overlap * weights_t_fade_in
    )

    t_dominant_start = rt_overlap_start + overlap_rt
    final_recon_x[:, :, t_dominant_start:] = recon_x_t[:, :, t_dominant_start:]

    return final_recon_x


def individual_test_vae(
    test_loader, vae, device, ts, ecg_ch_names, args, label_name, epoch
):
    vae.eval()
    loss_keep = 0
    mse_keep = 0
    kdl_keep = 0
    acc_keep = 0
    datalength = args.datalength
    with torch.no_grad():
        for j, (x, xo, label_name, pt_index) in enumerate(test_loader):
            x, xo = x.to(device), xo.to(device)
            recon_x, mean, log_var, z = vae(x)
            if args.loss_fn_type == "mse":
                if args.loss_pt_on_off == "off":
                    loss, mse, kdl = utils.loss_fn_mse(
                        recon_x, xo, mean, log_var, datalength, args
                    )
                elif args.loss_pt_on_off == "on":
                    loss, mse, kdl = utils.loss_fn_mse_pt(
                        recon_x,
                        xo,
                        mean,
                        log_var,
                        datalength,
                        pt_index=pt_index,
                        args=args,
                    )

            loss_keep += loss
            if args.loss_fn_type == "mse":
                mse_keep += mse
            kdl_keep += kdl
            mse_loss = torch.nn.MSELoss(reduction="mean")
            acc = mse_loss(recon_x.view(-1, datalength), xo.view(-1, datalength))
            acc_keep += acc
    return loss_keep


def all_test_vae(test_loader, vae_dict, device, ts, ecg_ch_names, args):
    vae_dict["P"].eval()
    vae_dict["R"].eval()
    vae_dict["T"].eval()

    label_temps = []
    test_val_all = []
    test_val_12ch_all = []
    test_val_all_rmse = []
    test_val_12ch_all_rmse = []
    pearson_scores = []
    ecg_ch = args.ecg_ch_num
    datalength = args.datalength

    with torch.no_grad():
        for j, (x, xo, label_name, pt_index) in enumerate(test_loader):
            x, xo = x.to(device), xo.to(device)
            recon_x_p, mean_p, log_var_p, z_p = vae_dict["P"](x)
            recon_x_r, mean_r, log_var_r, z_r = vae_dict["R"](x)
            recon_x_t, mean_t, log_var_t, z_t = vae_dict["T"](x)

            p_end_original = int(0.325 * datalength)
            r_start_original = int(0.425 * datalength)
            r_end_original = int(0.525 * datalength)
            t_start_original = int(0.525 * datalength)
            overlap_duration_pr = int(0.05 * datalength)
            overlap_duration_rt = int(0.05 * datalength)

            recon_x = weighted_average_join(
                recon_x_p,
                recon_x_r,
                recon_x_t,
                p_end_orig=p_end_original,
                r_start_orig=r_start_original,
                r_end_orig=r_end_original,
                t_start_orig=t_start_original,
                overlap_pr=overlap_duration_pr,
                overlap_rt=overlap_duration_rt,
            )

            for i in range(len(label_name)):
                label_temps.append(label_name[i])

            recon_x = recon_x.view(-1, ecg_ch, datalength)
            xo = xo.view(-1, ecg_ch, datalength)

            mae_loss = utils.MAE_2(reduction="none")
            acc_mae = mae_loss(recon_x, xo)

            if args.loss_pt_on_off == "off":
                test_val = utils.cul_val_no_pt(acc_mae)
            else:
                test_val = utils.cul_val(pt_index, acc_mae)

            test_val = [array_item.item() for array_item in test_val]
            test_val = np.array(test_val, dtype=np.float32)
            test_val_all += test_val.tolist()

            if args.loss_pt_on_off == "off":
                test_val_12ch = utils.cul_val_per_12ch_no_pt(acc_mae)
            else:
                test_val_12ch = utils.cul_val_per_12ch(pt_index, acc_mae)

            test_val_12ch = np.array(
                [array_item.tolist() for array_item in test_val_12ch],
                dtype=np.float32,
            )
            test_val_12ch = test_val_12ch.reshape(-1, args.ecg_ch_num)
            test_val_12ch_all += test_val_12ch.tolist()

            mse_loss = torch.nn.MSELoss(reduction="none")
            acc_mse = mse_loss(recon_x, xo)
            acc_rmse = torch.sqrt(acc_mse + 1e-8)

            if args.loss_pt_on_off == "off":
                test_val_rmse = utils.cul_val_no_pt(acc_rmse)
            else:
                test_val_rmse = utils.cul_val(pt_index, acc_rmse)

            test_val_rmse = [array_item.item() for array_item in test_val_rmse]
            test_val_rmse = np.array(test_val_rmse, dtype=np.float32)
            test_val_all_rmse += test_val_rmse.tolist()

            if args.loss_pt_on_off == "off":
                test_val_12ch_rmse = utils.cul_val_per_12ch_no_pt(acc_rmse)
            else:
                test_val_12ch_rmse = utils.cul_val_per_12ch(pt_index, acc_rmse)

            test_val_12ch_rmse = np.array(
                [array_item.tolist() for array_item in test_val_12ch_rmse],
                dtype=np.float32,
            )
            test_val_12ch_rmse = test_val_12ch_rmse.reshape(-1, args.ecg_ch_num)
            test_val_12ch_all_rmse += test_val_12ch_rmse.tolist()

            recon_x_np = recon_x.view(-1, datalength).cpu().numpy().astype(np.float64)
            xo_np = xo.view(-1, datalength).cpu().numpy().astype(np.float64)
            r, _ = utils.pearsonr(recon_x_np.ravel(), xo_np.ravel())
            pearson_scores.append(r)

            batch_size_now = xo.shape[0]

            utils.plot_fig_test_name_8ch_2row(
                recon_x=recon_x,
                xo=xo,
                datalength=datalength,
                ts=ts,
                args=args,
                batch_size_num=batch_size_now,
                label_name=label_name,
                acc=test_val_12ch,
                pt_index=pt_index,
                ecg_ch_names=ecg_ch_names,
            )

            utils.save_csv2(
                data=recon_x,
                args=args,
                ts=ts,
                label_name=label_name,
                data_rec_or_xo="recon_x",
            )
            utils.save_csv2(
                data=xo,
                args=args,
                ts=ts,
                label_name=label_name,
                data_rec_or_xo="xo",
            )

    pearson_score = sum(pearson_scores) / len(pearson_scores)

    test_val_all = np.array(test_val_all)
    test_val_mean = np.mean(test_val_all)
    test_val_12ch_all = np.array(test_val_12ch_all)
    test_val_12ch_mean = np.mean(test_val_12ch_all, axis=0)

    test_val_all_rmse = np.array(test_val_all_rmse)
    test_val_rmse_mean = np.mean(test_val_all_rmse)
    test_val_12ch_all_rmse = np.array(test_val_12ch_all_rmse)
    test_val_12ch_rmse_mean = np.mean(test_val_12ch_all_rmse, axis=0)

    data_to_write = {
        "TARGET_NAME": args.TARGET_NAME,
        "MAE_all": test_val_mean,
        "MAE_A1": test_val_12ch_mean[0],
        "MAE_A2": test_val_12ch_mean[1],
        "MAE_V1": test_val_12ch_mean[2],
        "MAE_V2": test_val_12ch_mean[3],
        "MAE_V3": test_val_12ch_mean[4],
        "MAE_V4": test_val_12ch_mean[5],
        "MAE_V5": test_val_12ch_mean[6],
        "MAE_V6": test_val_12ch_mean[7],
        "RMSE_all": test_val_rmse_mean,
        "RMSE_A1": test_val_12ch_rmse_mean[0],
        "RMSE_A2": test_val_12ch_rmse_mean[1],
        "RMSE_V1": test_val_12ch_rmse_mean[2],
        "RMSE_V2": test_val_12ch_rmse_mean[3],
        "RMSE_V3": test_val_12ch_rmse_mean[4],
        "RMSE_V4": test_val_12ch_rmse_mean[5],
        "RMSE_V5": test_val_12ch_rmse_mean[6],
        "RMSE_V6": test_val_12ch_rmse_mean[7],
        "pearson_score": pearson_score,
    }
    return data_to_write


def main():
    utils.create_directory_if_not_exists(os.path.join(base_dir, "model_pth"))
    args = config.get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    writer_P = SummaryWriter(
        log_dir=f"runs/{args.current_time}/{args.TARGET_NAME}/P_log"
    )
    writer_R = SummaryWriter(
        log_dir=f"runs/{args.current_time}/{args.TARGET_NAME}/R_log"
    )
    writer_T = SummaryWriter(
        log_dir=f"runs/{args.current_time}/{args.TARGET_NAME}/T_log"
    )
    writer_dict = {
        "P": writer_P,
        "R": writer_R,
        "T": writer_T,
    }

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    train_dataset_dict = {}
    test_dataset_dict = {}

    if args.Dataset_name == "15ch_arrange_direction":
        dataset_setup_fn = Dataset.Dataset_setup_8ch_pt_augmentation
        dataset_args = {
            "TARGET_NAME": args.TARGET_NAME,
            "transform_type": args.transform_type,
            "Dataset_name": args.Dataset_name,
            "dataset_num": args.dataset_num,
            "ave_data_flg": args.ave_data_flg,
            "datalength": args.datalength,
            "num_channels": args.num_channels,
        }
    else:
        dataset_setup_fn = Dataset.Dataset_setup_virtual_9ch
        dataset_args = {
            "TARGET_NAME": args.TARGET_NAME,
            "transform_type": args.transform_type,
            "Dataset_name": args.Dataset_name,
            "dataset_num": args.dataset_num,
            "ave_data_flg": args.ave_data_flg,
            "orientation": args.orientation,
            "datalength": args.datalength,
            "num_channels": args.num_channels,
        }

    train_dataset_dict["P_train_dataset"], test_dataset_dict["P_test_dataset"] = (
        dataset_setup_fn(**dataset_args, DataAugumentation=args.p_augumentation)
    )
    train_dataset_dict["R_train_dataset"], test_dataset_dict["R_test_dataset"] = (
        dataset_setup_fn(**dataset_args, DataAugumentation=args.r_augumentation)
    )
    train_dataset_dict["T_train_dataset"], test_dataset_dict["T_test_dataset"] = (
        dataset_setup_fn(**dataset_args, DataAugumentation=args.t_augumentation)
    )
    all_test_dataset = test_dataset_dict["R_test_dataset"]

    ts = (
        datetime.datetime.fromtimestamp(time.time()).strftime("%Y%m%d_%H%M%S")
        + "_Dataset_name="
        + args.Dataset_name
        + "_transform_type="
        + args.transform_type
        + "_"
        + args.mode
        + "_TARGETNAME="
        + str(args.TARGET_NAME)
        + "_beta="
        + str(args.beta)
        + "_alpha="
        + str(args.alpha)
        + "_loss_pt_on_off="
        + args.loss_pt_on_off
        + "_augument="
        + args.p_augumentation
        + "_"
        + args.r_augumentation
        + "_"
        + args.t_augumentation
    )

    utils.create_directory_if_not_exists(os.path.join(args.fig_root, str(ts)))

    with open(os.path.join(args.fig_root, str(ts), "args.json"), mode="w") as f:
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
        "num_channels": args.num_channels,  # Use num_channels arg
    }
    vae_dict = {
        "P": VAE(**common_kwargs).to(device),
        "R": VAE(**common_kwargs).to(device),
        "T": VAE(**common_kwargs).to(device),
    }

    if args.mode == "train":
        print("TRAINING MODE::\n")

        weights = {
            "P": args.loss_pt_on_off_P_weight,
            "R": args.loss_pt_on_off_R_weight,
            "T": args.loss_pt_on_off_T_weight,
        }
        weight_keys = list(weights.keys())
        for target_weight in weight_keys:
            current_weight = {}
            for weight_key in weight_keys:
                left, right = weights[weight_key].split()
                current_weight[weight_key] = (
                    float(right) if target_weight == weight_key else float(left)
                )
            train_data_loader = DataLoader(
                dataset=train_dataset_dict[f"{target_weight}_train_dataset"],
                batch_size=args.train_batch_size,
                shuffle=True,
            )

            test_loader = DataLoader(
                test_dataset_dict[f"{target_weight}_test_dataset"],
                batch_size=4,
                shuffle=False,
            )
            vae = vae_dict[target_weight]
            optimizer = torch.optim.Adam(vae.parameters(), lr=args.learning_rate)

            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.1, patience=5, verbose=True
            )

            for epoch in range(args.epochs):
                vae.train()
                loss_keep = 0.0
                mse_keep = 0.0
                kdl_keep = 0.0
                acc_keep = 0.0

                for iteration, (x, xo, label_name, pt_index) in enumerate(
                    train_data_loader
                ):
                    x, xo = x.to(device), xo.to(device)
                    recon_x, mean, log_var, z = vae(x)

                    if args.loss_fn_type == "mse":
                        if len(current_weight) == 3:
                            loss, mse, kdl = utils.loss_fn_mse_PRT(
                                recon_x,
                                xo,
                                mean,
                                log_var,
                                args.datalength,
                                args,
                                current_weight,
                                pt_index,
                            )
                        else:
                            loss, mse, kdl = utils.loss_fn_mse(
                                recon_x, xo, mean, log_var, args.datalength, args
                            )

                    loss_keep += loss
                    if args.loss_fn_type == "mse":
                        mse_keep += mse
                    kdl_keep += kdl
                    mse_loss = torch.nn.MSELoss(reduction="mean")
                    acc = mse_loss(
                        recon_x.view(-1, args.datalength), xo.view(-1, args.datalength)
                    )
                    acc_keep += acc

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                current_lr = scheduler.get_last_lr()[0]

                print(
                    "Epoch {:02d}/{:02d} Batch {:04d}/{:d}, Loss {:9.4f}, Acc(mse) {:9.4f}, LR {:9.6f}".format(
                        epoch,
                        args.epochs,
                        iteration,
                        len(train_data_loader) - 1,
                        loss_keep.item() / (iteration + 1),
                        acc_keep.item() / (iteration + 1),
                        current_lr,
                    )
                )

                writer_dict[target_weight].add_scalar(
                    "Loss/train_loss", loss_keep.item() / len(train_data_loader), epoch
                )
                writer_dict[target_weight].add_scalar(
                    "Learning Rate", current_lr, epoch
                )
                writer_dict[target_weight].add_scalar(
                    "KL Divergence",
                    kdl_keep.item() / len(train_data_loader),
                    epoch,
                )
                writer_dict[target_weight].add_scalar(
                    "MSE Loss",
                    mse_keep / len(train_data_loader),
                    epoch,
                )
                test_loss = individual_test_vae(
                    test_loader,
                    vae,
                    device,
                    ts,
                    [],
                    args,
                    None,
                    epoch,
                )
                writer_dict[target_weight].add_scalar(
                    "Loss/test_loss", test_loss.item() / len(test_loader), epoch
                )

            base_path = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "..")
            )
            if target_weight == "P":
                torch.save(
                    vae.state_dict(),
                    os.path.join(base_path, "model_pth", "vae_pwave_weight.pth"),
                )
                torch.save(
                    vae.state_dict(),
                    os.path.join(args.fig_root, str(ts), "vae_pwave_weight.pth"),
                )
            elif target_weight == "R":
                torch.save(
                    vae.state_dict(),
                    os.path.join(base_path, "model_pth", "vae_rwave_weight.pth"),
                )
                torch.save(
                    vae.state_dict(),
                    os.path.join(args.fig_root, str(ts), "model_rwave_weight.pth"),
                )
            elif target_weight == "T":
                torch.save(
                    vae.state_dict(),
                    os.path.join(base_path, "model_pth", "vae_twave_weight.pth"),
                )
                torch.save(
                    vae.state_dict(),
                    os.path.join(args.fig_root, str(ts), "vae_twave_weight.pth"),
                )

            vae_dict[target_weight] = vae

        data_to_write = all_test_vae(test_loader, vae_dict, device, ts, [], args)
        output_file = os.path.join(
            args.mae_folder
            + "/MAE_leave_1_out_{}_PRTweight_{}_{}_{}_augumentation={}.csv".format(
                args.Dataset_name,
                str(args.loss_pt_on_off_P_weight),
                str(args.loss_pt_on_off_R_weight),
                str(args.loss_pt_on_off_T_weight),
                args.p_augumentation,
                args.r_augumentation,
                args.t_augumentation,
            )
        )
        utils.write_to_csv(output_file, data=data_to_write)
        writer_dict[target_weight].close()

    elif args.mode == "test":
        print("TEST MODE::\n")
        test_loader = DataLoader(all_test_dataset, batch_size=4, shuffle=False)
        all_test_vae(test_loader, vae_dict, ts, [], args)


if __name__ == "__main__":
    main()
