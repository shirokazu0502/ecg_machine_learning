import math
import os
import re
import sys
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import codecs
import datetime
from scipy import signal
from scipy.interpolate import interp1d
import matplotlib.cm as cm
import matplotlib
from scipy.stats import pearsonr
import pandas as pd
import csv
import neurokit2 as nk

from settings import (
    DATA_DIR,
    BASE_DIR,
    PROCESSED_DATA_DIR,
    OUTPUT_DIR,
    RAW_DATA_DIR,
    TEST_DIR,
    RATE,
    RATE_15CH,
    TIME,
    DATASET_MADE_DATE,
    OUTPUT_MAE_DIR,
)

cmap = "tab10"


def extract_between_third_and_fourth_underscore(input_string):
    parts = input_string.split("_")
    if len(parts) >= 4:
        result = parts[3]
        return result
    else:
        return None


def hpf(d_in, sampling_rate, fp, fs):
    gpass = 1
    gstop = 40
    norm_pass = fp / (sampling_rate / 2)
    norm_stop = fs / (sampling_rate / 2)
    N, Wn = signal.cheb2ord(
        wp=norm_pass, ws=norm_stop, gpass=gpass, gstop=gstop, analog=0
    )
    b, a = signal.cheby2(N, gstop, Wn, "high")
    d_out = signal.filtfilt(b, a, d_in)
    return d_out


def lpf(d_in, sampling_rate, fp, fs):
    gpass = 1
    gstop = 40
    norm_pass = fp / (sampling_rate / 2)
    norm_stop = fs / (sampling_rate / 2)
    N, Wn = signal.cheb2ord(
        wp=norm_pass, ws=norm_stop, gpass=gpass, gstop=gstop, analog=0
    )
    b, a = signal.cheby2(N, gstop, Wn, "low")
    d_out = signal.filtfilt(b, a, d_in)
    return d_out


def min_max_old(x):
    min_val = x.min(axis=None, keepdims=True)
    max_val = x.max(axis=None, keepdims=True)
    if (max_val - min_val) != 0:
        result = (x - min_val) / (max_val - min_val)
        return result
    else:
        return x * 0


def min_max_2(x):
    num = x.shape[0]
    for i in range(num):
        min_val = x[i].min(axis=None, keepdims=True)
        max_val = x[i].max(axis=None, keepdims=True)
        if (max_val - min_val) != 0:
            a = max(abs(max_val), abs(min_val))
            x[i] = x[i] / (2.0 * a) + 0.5
    return x


def min_max(x, minth, maxth):
    if (maxth - minth) != 0:
        result = (x - minth) / (maxth - minth)
        return np.clip(result, 0, 1.0)
    else:
        return x * 0


def plot_fig(
    numplotfig, recon_x, xo, datalength, exp_dir, args, label_name, ecg_ch_names
):
    sample_rate = 500
    sample_num = datalength
    xticks = np.linspace(0.0, 1.0 / sample_rate * sample_num, sample_num)
    ecg_ch = args.ecg_ch_num
    for p in range(numplotfig):
        for q in range(ecg_ch):
            recon_x2 = torch.reshape(recon_x, (-1, ecg_ch, datalength))
            xo2 = torch.reshape(xo, (-1, ecg_ch, datalength))
            plt.rcParams["font.size"] = 16
            plt.rcParams["xtick.direction"] = "in"
            plt.rcParams["ytick.direction"] = "in"
            plt.plot(
                xticks,
                recon_x2[p][q].cpu().data.numpy(),
                color="red",
                linewidth=1.0,
                linestyle="-",
            )
            plt.plot(
                xticks,
                xo2[p][q].cpu().data.numpy(),
                color="blue",
                linewidth=1.0,
                linestyle="-",
            )
            plt.xlim(0.0, sample_num / sample_rate)
            plt.xlabel("second")
            plt.ylabel("amplitude")
            plt.axis("on")
            plt.minorticks_on()
            plt.grid(which="both", axis="x", alpha=0.8, linestyle="--", linewidth=1)
            plt.legend(
                ["predict", "ECG"],
                bbox_to_anchor=(0.60, 1),
                loc="upper left",
                fontsize=16,
                framealpha=1.0,
            )
            plt.title(label_name[p] + "{}".format(ecg_ch_names[q]))
            plt.tight_layout()
            save_path = os.path.join(exp_dir, "reconstructed_plots")
            os.makedirs(save_path, exist_ok=True)
            plt.savefig(
                os.path.join(save_path, f"test_{label_name[p]}_ch{q}.png"),
                dpi=300,
            )
            plt.cla()
            plt.clf()
            plt.close()


def pearson_corr_loss_like_scipy(recon_x, x):
    recon_x = recon_x.view(recon_x.size(0), -1)
    x = x.view(x.size(0), -1)
    losses = []
    for i in range(recon_x.size(0)):
        vx = recon_x[i] - torch.mean(recon_x[i])
        vy = x[i] - torch.mean(x[i])
        numerator = torch.sum(vx * vy)
        denominator = torch.sqrt(torch.sum(vx**2)) * torch.sqrt(torch.sum(vy**2)) + 1e-8
        r = numerator / denominator
        losses.append(1 - r)
    return torch.mean(torch.stack(losses))


def loss_fn_mse(recon_x, x):
    criterion_mse = torch.nn.MSELoss(reduction="mean")
    mse_loss = criterion_mse(recon_x, x)
    print(f"mse_loss: {mse_loss.item()}")
    return mse_loss


def loss_fn_mse_and_corr(recon_x, x, beta=0.2):
    criterion_mse = torch.nn.MSELoss(reduction="mean")
    mse_loss = criterion_mse(recon_x, x)
    corr_loss = pearson_corr_loss_like_scipy(recon_x, x)
    # --- NEW: Scale by order of magnitude (powers of 10) ---
    with torch.no_grad():
        epsilon = 1e-12  # To prevent log10(0) for very small losses
        # Get Python scalar values for log10 calculation
        mse_item = mse_loss.item()
        corr_item = corr_loss.item()
        # Calculate order of magnitude for each loss
        order_mse = math.floor(math.log10(mse_item + epsilon))
        order_corr = math.floor(math.log10(corr_item + epsilon))
        # Calculate the difference in order of magnitude
        order_diff = order_mse - order_corr
        # The scaling factor is 10 to the power of the difference
        scaling_factor = 10.0**order_diff
    # Apply the quantized scaling factor
    scaled_corr_loss = corr_loss * scaling_factor
    combined_loss = mse_loss + beta * scaled_corr_loss
    return combined_loss, mse_loss, scaled_corr_loss


def loss_fn_weighted_mse_and_corr(recon_x, x, pt_index, beta, weight_r, weight_other):
    """
    Calculates a combined loss of weighted MSE and scaled Pearson correlation.
    The MSE is weighted differently for the R-wave segment vs. other segments.
    """
    # 1. Calculate Raw Pearson Correlation Loss
    raw_corr_loss = pearson_corr_loss_like_scipy(recon_x, x)

    # 2. Calculate Weighted MSE
    batch_size, num_channels, datalength = x.shape
    total_weighted_mse = torch.tensor(0.0, device=x.device)

    mse_func_sum = torch.nn.MSELoss(
        reduction="sum"
    )  # Use sum reduction for per-sample MSE

    for i in range(batch_size):
        # pt_index: [P_onset, P_offset, QRS_onset, QRS_offset, T_onset, T_offset, heart_rate, heart_rate_variability]
        # We need QRS_onset (index 2) and QRS_offset (index 3)
        # pt_index is (batch_size, 8)
        qrs_onset = int(pt_index[i, 2].item())
        qrs_offset = int(pt_index[i, 3].item())

        # Ensure indices are within bounds
        if qrs_onset < 0:
            qrs_onset = 0
        if qrs_offset > datalength:
            qrs_offset = datalength

        # Handle cases where QRS is invalid or too short
        if qrs_onset >= qrs_offset:
            # Fall back to unweighted MSE for this sample if R-wave segment is invalid
            total_weighted_mse += mse_func_sum(recon_x[i], x[i])
            continue

        # Extract R-wave segment and other segments
        recon_r = recon_x[i, :, qrs_onset:qrs_offset]
        target_r = x[i, :, qrs_onset:qrs_offset]

        # Concatenate non-R-wave segments
        # Handle cases where segments before or after QRS might be empty
        recon_other_parts = []
        target_other_parts = []

        if qrs_onset > 0:
            recon_other_parts.append(recon_x[i, :, :qrs_onset])
            target_other_parts.append(x[i, :, :qrs_onset])

        if qrs_offset < datalength:
            recon_other_parts.append(recon_x[i, :, qrs_offset:])
            target_other_parts.append(x[i, :, qrs_offset:])

        if (
            not recon_other_parts
        ):  # If no 'other' parts (e.g., R-wave covers entire segment)
            mse_other = torch.tensor(0.0, device=x.device)
        else:
            recon_other = torch.cat(recon_other_parts, dim=1)
            target_other = torch.cat(target_other_parts, dim=1)
            mse_other = mse_func_sum(recon_other, target_other)

        # Calculate MSE for R-wave segment
        mse_r = mse_func_sum(recon_r, target_r)

        # Apply individual weights and sum for this sample
        weighted_mse_sample = (weight_r * mse_r) + (weight_other * mse_other)
        total_weighted_mse += weighted_mse_sample

    # Average the weighted MSE over the batch and normalize by total elements
    # mse_func_sum calculates sum of squared diff, need to divide by num_elements
    weighted_mse_loss = total_weighted_mse / (batch_size * num_channels * datalength)

    # 3. Scale Raw Pearson Correlation Loss to match Weighted MSE order
    with torch.no_grad():
        epsilon = 1e-12
        order_mse = math.floor(math.log10(weighted_mse_loss.item() + epsilon))
        order_corr = math.floor(math.log10(raw_corr_loss.item() + epsilon))
        order_diff = order_mse - order_corr
        scaling_factor = 10.0**order_diff

    scaled_corr_loss = raw_corr_loss * scaling_factor

    # 4. Combine the weighted MSE and scaled Pearson correlation loss
    combined_loss = weighted_mse_loss + beta * scaled_corr_loss

    return combined_loss


def loss_fn_mse_PRT(
    recon_x, x, mean, log_var, datalength, args, current_weight, pt_index
):
    weight_P = current_weight["P"]
    weight_R = current_weight["R"]
    weight_T = current_weight["T"]
    Q_peaks = pt_index[:, 5]
    S_peaks = pt_index[:, 6]
    batch_size = x.shape[0]
    MSE = torch.nn.MSELoss(reduction="sum")
    MSE_loss = 0.0
    for i in range(batch_size):
        index1 = 120
        index2 = 180
        recon_x_before_R1 = recon_x.view(-1, args.ecg_ch_num, datalength)[i, :, :index1]
        recon_x_after_R2 = recon_x.view(-1, args.ecg_ch_num, datalength)[i, :, index2:]
        x_before_R1 = x.view(-1, args.ecg_ch_num, datalength)[i, :, :index1]
        x_after_R2 = x.view(-1, args.ecg_ch_num, datalength)[i, :, index2:]
        recon_x_R = recon_x.view(-1, args.ecg_ch_num, datalength)[i, :, index1:index2]
        x_R = x.view(-1, args.ecg_ch_num, datalength)[i, :, index1:index2]
        MSE_loss_1 = MSE(recon_x_before_R1, x_before_R1)
        MSE_loss_2 = MSE(recon_x_after_R2, x_after_R2)
        MSE_loss_R = MSE(recon_x_R, x_R)
        MSE_loss_keep = (
            MSE_loss_1 * weight_P + MSE_loss_2 * weight_T + MSE_loss_R * weight_R
        )
        MSE_loss = MSE_loss + MSE_loss_keep
    MSE_loss = MSE_loss / args.ecg_ch_num / datalength / batch_size
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    alpha = args.alpha
    beta = args.beta
    return (
        (alpha * MSE_loss + beta * KLD / batch_size),
        alpha * MSE_loss,
        beta * KLD / batch_size,
    )


def loss_fn_mse_kld(recon_x, x, mean, log_var, datalength, args):
    batch_size = x.shape[0]
    # print(x.shape)
    # print(recon_x.shape)
    # MSE=torch.nn.MSELoss(reduction="sum")#reducitonをsumにしていた。これはミニバッチの全ての要素の二乗誤差の和。meanのときの要素数倍になる。
    MSE = torch.nn.MSELoss(reduction="mean")
    MSE_loss = MSE(
        recon_x.view(-1, datalength), x.view(-1, datalength)
    )  # ミニバッチ内全てで平均。ミニバッチ内のそれぞれのデータの形状が同じ
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    alpha = args.alpha
    beta = args.beta
    # print(MSE_loss)
    # # print("pure_MSE")
    # KLD=beta*KLD
    # return (alpha*MSE_loss + beta*KLD) / batch_size, MSE_loss/batch_size, KLD/batch_size
    return (
        alpha * MSE_loss + beta * KLD / batch_size,
        MSE_loss / batch_size,
        KLD / batch_size,
    )  # MSEはreduction=meanで既にバッチで割られているからここではバッチ数で割らない
    # return (alpha*MSE_loss + KLD) / x.size(0), MSE_loss/x.size(0), KLD/x.size(0)


def noise_make(mean, scale, datanum, ch_num):
    rnd = np.random.normal(loc=mean, scale=scale, size=datanum * ch_num)
    rnd = rnd.reshape(-1, ch_num, datanum)


def tensor_to_ndarray(tensor):
    if isinstance(tensor, torch.Tensor):
        z = tensor.detach().cpu().numpy()
        return z
    else:
        raise TypeError("Input must be a torch.Tensor.")


def extract_pos_name(string):
    pattern = r"\w+_\w+_\w+_(\w+)_\w+"
    match = re.search(pattern, string)
    if match:
        last_name = match.group(1)
        return last_name
    else:
        return None


def extract_person_name(string):
    pattern = r"(\w+)_\w+"
    match = re.search(pattern, string)
    if match:
        last_name = match.group(1)
        return last_name
    else:
        return None


class RMSELoss(torch.nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.mse = torch.nn.MSELoss(reduction=reduction)

    def forward(self, yhat, y):
        return torch.sqrt(self.mse(yhat, y))


class MAE(torch.nn.Module):
    def __init__(self, reduction="none"):
        super().__init__()
        self.mse = torch.nn.MSELoss(reduction=reduction)

    def forward(self, yhat, y):
        return torch.sqrt(self.mse(yhat, y))


class MAE_2(torch.nn.Module):
    def __init__(self, reduction="none"):
        super().__init__()
        self.l1_loss = torch.nn.L1Loss(reduction=reduction)

    def forward(self, yhat, y):
        return self.l1_loss(yhat, y)


def cul_val_no_pt(acc):
    batch_size = acc.shape[0]
    acc_list = []
    for i in range(batch_size):
        acc_pt_12ch = torch.mean(acc[i, :, :], dim=(0, 1))
        acc_pt_12ch = acc_pt_12ch.to("cpu").detach().numpy()
        acc_list.append(acc_pt_12ch)
    return acc_list


def cul_val(pt, acc):
    batch_size = acc.shape[0]
    acc_list = []
    for i in range(batch_size):
        acc_pt_12ch = torch.mean(acc[i, :, pt[i, 0] : pt[i, 1]], dim=(0, 1))
        acc_pt_12ch = acc_pt_12ch.to("cpu").detach().numpy()
        acc_list.append(acc_pt_12ch)
    return acc_list


def cul_val_per_12ch_no_pt(acc):
    batch_size = acc.shape[0]
    acc_list = []
    for i in range(batch_size):
        acc_pt_12ch = torch.mean(acc[i, :, :], dim=(1))
        acc_pt_12ch = acc_pt_12ch.to("cpu").detach().numpy()
        acc_list.append(acc_pt_12ch)
    return acc_list


def cul_val_per_12ch(pt, acc):
    batch_size = acc.shape[0]
    acc_list = []
    for i in range(batch_size):
        acc_pt_12ch = torch.mean(acc[i, :, pt[i, 0] : pt[i, 1]], dim=(1))
        acc_pt_12ch = acc_pt_12ch.to("cpu").detach().numpy()
        acc_list.append(acc_pt_12ch)
    return acc_list


def write_to_csv(file_path, data):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)  # 追加
    file_exists = os.path.exists(file_path)
    with open(file_path, "a", newline="") as csvfile:
        fieldnames = [
            "TARGET_NAME",
            "MAE_all",
            "MAE_A1",
            "MAE_A2",
            "MAE_V1",
            "MAE_V2",
            "MAE_V3",
            "MAE_V4",
            "MAE_V5",
            "MAE_V6",
            "RMSE_all",
            "RMSE_A1",
            "RMSE_A2",
            "RMSE_V1",
            "RMSE_V2",
            "RMSE_V3",
            "RMSE_V4",
            "RMSE_V5",
            "RMSE_V6",
            "pearson_all",
            "pearson_A1",
            "pearson_A2",
            "pearson_V1",
            "pearson_V2",
            "pearson_V3",
            "pearson_V4",
            "pearson_V5",
            "pearson_V6",
            "Relative_Roughness",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(data)


def save_csv2(data, args, exp_dir, label_name, data_rec_or_xo):
    data = torch.reshape(data, (-1, 8, args.datalength))
    data_np = data.cpu().numpy() if data.is_cuda else data.numpy()
    assert len(data.shape) == 3, "Input data shape is incorrect."
    output_path = os.path.join(exp_dir, "waveforms")
    os.makedirs(output_path, exist_ok=True)
    batch_size = data_np.shape[0]
    for p in range(batch_size):
        if data_rec_or_xo == "recon_x":
            output_file = os.path.join(
                output_path, "{}_reconx.csv".format(label_name[p])
            )
        else:
            output_file = os.path.join(output_path, "{}_xo.csv".format(label_name[p]))
        df_data = pd.DataFrame(data_np[p])
        df_data = df_data.T
        new_columns = ["A1", "A2", "V1", "V2", "V3", "V4", "V5", "V6"]
        df_data.columns = new_columns
        df_data.to_csv(output_file, index=None)
    return output_path


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
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def create_directory_if_not_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)


def plot_fig_test_name_8ch_2row(
    recon_x,
    xo,
    datalength,
    exp_dir,
    args,
    batch_size_num,
    label_name,
    acc,
    pt_index,
    ecg_ch_names,
):
    sample_rate = 500
    sample_num = datalength
    xticks = np.linspace(0.0, 1.0 / sample_rate * sample_num, sample_num)
    ecg_ch = args.ecg_ch_num
    recon_x_reshaped = torch.reshape(recon_x, (-1, ecg_ch, datalength))
    xo_reshaped = torch.reshape(xo, (-1, ecg_ch, datalength))

    for p in range(batch_size_num):
        fig, axs = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle(f"{label_name[p]}", fontsize=20)

        for q in range(ecg_ch):
            row = q // 4
            col = q % 4
            ax = axs[row, col]

            recon_data = recon_x_reshaped[p][q].cpu().data.numpy()
            xo_data = xo_reshaped[p][q].cpu().data.numpy()

            ax.plot(
                xticks,
                recon_data,
                color="red",
                linewidth=1.0,
                linestyle="-",
                label="predict",
            )
            ax.plot(
                xticks, xo_data, color="blue", linewidth=1.0, linestyle="-", label="ECG"
            )

            ax.set_xlim(0.0, sample_num / sample_rate)
            ax.set_xlabel("second")
            ax.set_ylabel("amplitude")
            ax.grid(which="both", axis="x", alpha=0.8, linestyle="--", linewidth=1)
            ax.set_title(f"{ecg_ch_names[q]}")
            ax.legend()

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        save_path = os.path.join(exp_dir, "reconstructed_plots")
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(
            os.path.join(save_path, f"test_all_channel_{label_name[p]}.svg"),
            dpi=300,
        )
        plt.cla()
        plt.clf()
        plt.close(fig)


def total_variation_loss(y_pred, weight=1.0):
    # ... (existing function) ...
    return weight * tv_loss / batch_size


def smoothness_loss(recon_x, weight=0.1):
    # Calculate the L2 norm of the first derivative (difference between adjacent points)
    # recon_x shape: (batch_size, num_channels, datalength)
    diff = recon_x[:, :, 1:] - recon_x[:, :, :-1]
    smooth_loss = torch.mean(diff**2)
    return weight * smooth_loss


def loss_fn_lstm(recon_x, x, smoothness_weight=0.1):
    mse_loss = torch.nn.MSELoss(reduction="mean")(recon_x, x)
    s_loss = smoothness_loss(recon_x, weight=smoothness_weight)
    return mse_loss + s_loss


def calculate_relative_roughness(recon_x, xo):
    """
    Calculates the relative roughness of the reconstructed signal compared to the original.
    Roughness is defined as the mean absolute value of the second derivative.
    """
    # Calculate second derivative for recon_x
    diff1_recon = recon_x[:, :, 1:] - recon_x[:, :, :-1]
    diff2_recon = diff1_recon[:, :, 1:] - diff1_recon[:, :, :-1]
    roughness_recon = torch.mean(torch.abs(diff2_recon))

    # Calculate second derivative for xo
    diff1_xo = xo[:, :, 1:] - xo[:, :, :-1]
    diff2_xo = diff1_xo[:, :, 1:] - diff1_xo[:, :, :-1]
    roughness_xo = torch.mean(torch.abs(diff2_xo))

    # Calculate relative roughness, adding epsilon to avoid division by zero
    relative_roughness = roughness_recon / (roughness_xo + 1e-8)

    return relative_roughness.item()


def extract_wdd_features(waveform, sampling_rate=500):
    """
    Extracts diagnostically relevant features from a single ECG waveform using NeuroKit2.
    Input waveform should be a 1D NumPy array.
    """
    if waveform is None or len(waveform) == 0:
        return None

    try:
        # Process the ECG signal to find R-peaks
        _, rpeaks = nk.ecg_peaks(waveform, sampling_rate=sampling_rate)
        if rpeaks["ECG_R_Peaks"].size == 0:
            # If no R-peaks are found, delineation is not possible
            return None

        # Delineate the ECG signal to find P, Q, S, T waves
        _, waves = nk.ecg_delineate(
            waveform, rpeaks, sampling_rate=sampling_rate, method="dwt"
        )
    except Exception:
        # Neurokit can sometimes fail on unusual waveforms
        return None

    features = {}

    # Amplitudes
    print(f"waves: {waves}")
    if any(waves["ECG_P_Peaks"]) and not np.isnan(waves["ECG_P_Peaks"]).all():
        features["p_amp"] = np.nanmean(waveform[waves["ECG_P_Peaks"]])
    # if any(waves["ECG_R_Peaks"]) and not np.isnan(waves["ECG_R_Peaks"]).all():
    #     features["r_amp"] = np.nanmean(waveform[waves["ECG_R_Peaks"]])
    if any(waves["ECG_T_Peaks"]) and not np.isnan(waves["ECG_T_Peaks"]).all():
        features["t_amp"] = np.nanmean(waveform[waves["ECG_T_Peaks"]])

    # Durations (in milliseconds)
    ms_per_sample = 1000.0 / sampling_rate
    if any(waves["ECG_P_Onsets"]) and any(waves["ECG_P_Offsets"]):
        p_durations = (
            np.array(waves["ECG_P_Offsets"]) - np.array(waves["ECG_P_Onsets"])
        ) * ms_per_sample
        features["p_duration"] = np.nanmean(p_durations)

    if any(waves["ECG_Q_Peaks"]) and any(waves["ECG_S_Peaks"]):
        qrs_durations = (
            np.array(waves["ECG_S_Peaks"]) - np.array(waves["ECG_Q_Peaks"])
        ) * ms_per_sample
        features["qrs_duration"] = np.nanmean(qrs_durations)

    if any(waves["ECG_T_Onsets"]) and any(waves["ECG_T_Offsets"]):
        qt_intervals = (
            np.array(waves["ECG_T_Offsets"]) - np.array(waves["ECG_Q_Peaks"])
        ) * ms_per_sample
        features["qt_interval"] = np.nanmean(qt_intervals)

    return features


def calculate_wdd(features_orig, features_recon, default_weights=None):
    """
    Calculates the Weighted Diagnostic Distortion (WDD) between two sets of features.
    """
    if features_orig is None or features_recon is None:
        return np.nan  # Cannot compare if one of the delineations failed

    if default_weights is None:
        # Default weights, can be replaced by weights from literature
        default_weights = {
            "p_amp": 1.0,
            "r_amp": 1.0,
            "t_amp": 1.0,
            "p_duration": 1.0,
            "qrs_duration": 1.0,
            "qt_interval": 1.0,
        }

    total_distortion = 0.0
    total_weight = 0.0
    epsilon = 1e-6

    all_feature_keys = set(features_orig.keys()) | set(features_recon.keys())

    for key in all_feature_keys:
        w = default_weights.get(key, 1.0)
        v_orig = features_orig.get(key)
        v_recon = features_recon.get(key)

        # Handle missing features - this is a key part of WDD
        if v_orig is None or np.isnan(v_orig):
            # Feature exists in recon but not in orig (False Positive)
            # Penalize heavily if the reconstructed feature has a non-zero value
            distortion = (
                w * (abs(v_recon) / (abs(v_recon) + epsilon))
                if v_recon is not None and not np.isnan(v_recon)
                else 0.0
            )
        elif v_recon is None or np.isnan(v_recon):
            # Feature exists in orig but not in recon (False Negative)
            # Penalize heavily if the original feature was significant
            distortion = w * (abs(v_orig) / (abs(v_orig) + epsilon))
        else:
            # Both features exist, calculate normalized difference
            distortion = w * (abs(v_orig - v_recon) / (abs(v_orig) + epsilon))

        total_distortion += distortion
        total_weight += w  # Keep track of total weight for normalization

    if total_weight == 0:
        return 0.0

    return total_distortion / total_weight


def add_gaussian_noise(signal, noise_level=0.05):
    """Adds Gaussian noise to a signal."""
    noise = np.random.normal(0, noise_level, signal.shape)
    return signal + noise


def scale_amplitude(signal, scale_range=(0.9, 1.1)):
    """Scales the amplitude of a signal by a random factor."""
    scale_factor = np.random.uniform(scale_range[0], scale_range[1])
    return signal * scale_factor


def add_baseline_wander(signal, wander_freq=0.05, wander_amplitude=0.1):
    """Adds baseline wander to a signal."""
    t = np.arange(len(signal))
    wander = wander_amplitude * np.sin(2 * np.pi * wander_freq * t / len(signal))
    return signal + wander


def linear_interpolation_All(extation_range_ECG, extation_range_PGV, extation_rate):
    # 時系列データの時間情報を正規化
    # print(extantion_range_ECG.shape[1])
    length = extation_range_ECG.shape[1]
    # print(extation_range_ECG)

    x = np.arange(length)
    new_x = np.linspace(0, length - 1, int((length) * extation_rate))
    ECG_shape = (extation_range_ECG.shape[0], len(new_x))
    # print(extation_range_ECG.shape)
    # print(ECG_shape)
    # input("")
    new_tensor_ECG = torch.zeros(ECG_shape, dtype=torch.float32)
    # print(new_tensor_ECG)
    PGV_shape = (extation_range_PGV.shape[0], len(new_x))
    new_tensor_PGV = torch.zeros(PGV_shape, dtype=torch.float32)
    for i in range(extation_range_ECG.shape[0]):
        data = extation_range_ECG[i, :].numpy().copy()
        interpolator = interp1d(x, data)
        new_data = interpolator(new_x)
        new_data_tensor_ECG = torch.tensor(new_data)
        new_tensor_ECG[i] = new_data_tensor_ECG
        # print(new_tensor_ECG[i])
    for i in range(extation_range_PGV.shape[0]):
        data = extation_range_PGV[i, :].numpy().copy()
        interpolator = interp1d(x, data)
        new_data = interpolator(new_x)
        new_data_tensor = torch.tensor(new_data)
        new_tensor_PGV[i] = new_data_tensor
    return new_tensor_ECG, new_tensor_PGV


def make_p_onset_extension_datas(
    PGV_datas, ECG_datas, pt_array, label_name, extation_rate
):
    p_offset_org = pt_array[2]
    r_onset = pt_array[6]  # Correctly use q_peak for PR interval end
    if p_offset_org >= r_onset:
        return ECG_datas, PGV_datas, label_name, pt_array

    extation_range_ECG = ECG_datas[:, p_offset_org:r_onset]
    extation_range_PGV = PGV_datas[:, p_offset_org:r_onset]

    if extation_range_ECG.shape[1] <= 0:
        return ECG_datas, PGV_datas, label_name, pt_array

    new_extation_range_ECG, new_extation_range_PGV = linear_interpolation_All(
        extation_range_ECG=extation_range_ECG,
        extation_range_PGV=extation_range_PGV,
        extation_rate=extation_rate,
    )
    new_ECG_data = torch.concat(
        [ECG_datas[:, :p_offset_org], new_extation_range_ECG, ECG_datas[:, r_onset:]],
        dim=1,
    )
    new_PGV_data = torch.concat(
        [PGV_datas[:, :p_offset_org], new_extation_range_PGV, PGV_datas[:, r_onset:]],
        dim=1,
    )
    slide_index = new_ECG_data.shape[1] - 400
    if new_ECG_data.shape[1] > 400:
        new_ECG_data_400 = new_ECG_data[:, (new_ECG_data.shape[1] - 400) :]
        new_PGV_data_400 = new_PGV_data[:, (new_ECG_data.shape[1] - 400) :]
    else:
        First_ECG_value_tensor = new_ECG_data[:, 0]
        First_ECG_value_tensor_view = First_ECG_value_tensor.view(
            First_ECG_value_tensor.shape[0], 1
        )
        First_ECG_value_view_tensors = torch.cat(
            [First_ECG_value_tensor_view] * (400 - new_ECG_data.shape[1]), dim=1
        )
        new_ECG_data_400 = torch.concat(
            [First_ECG_value_view_tensors, new_ECG_data], dim=1
        )
        First_PGV_value_tensor = new_PGV_data[:, -1]
        First_PGV_value_tensor_view = First_PGV_value_tensor.view(
            First_PGV_value_tensor.shape[0], 1
        )
        First_PGV_value_view_tensors = torch.cat(
            [First_PGV_value_tensor_view] * (400 - new_PGV_data.shape[1]), dim=1
        )
        new_PGV_data_400 = torch.concat(
            [First_PGV_value_view_tensors, new_PGV_data], dim=1
        )
    new_label_name = label_name + "extraction_P=" + str(extation_rate)
    pt_array_augumentation = pt_array.copy()
    pt_array_augumentation[0] = pt_array_augumentation[0] - slide_index
    pt_array_augumentation[2] = pt_array_augumentation[2] - slide_index
    return new_ECG_data_400, new_PGV_data_400, new_label_name, pt_array_augumentation


def make_t_onset_extension_datas(
    PGV_datas, ECG_datas, pt_array, label_name, extation_rate
):
    t_onset_org = pt_array[3]
    r_offset = pt_array[7]  # Use S-peak as the start of the ST segment
    if r_offset >= t_onset_org:
        return ECG_datas, PGV_datas, label_name, pt_array

    extation_range_ECG = ECG_datas[:, r_offset:t_onset_org]
    extation_range_PGV = PGV_datas[:, r_offset:t_onset_org]

    if extation_range_ECG.shape[1] <= 0:
        return ECG_datas, PGV_datas, label_name, pt_array

    new_extation_range_ECG, new_extation_range_PGV = linear_interpolation_All(
        extation_range_ECG=extation_range_ECG,
        extation_range_PGV=extation_range_PGV,
        extation_rate=extation_rate,
    )
    new_ECG_data = torch.concat(
        [ECG_datas[:, :r_offset], new_extation_range_ECG, ECG_datas[:, t_onset_org:]],
        dim=1,
    )
    new_PGV_data = torch.concat(
        [PGV_datas[:, :r_offset], new_extation_range_PGV, PGV_datas[:, t_onset_org:]],
        dim=1,
    )
    slide_index = new_ECG_data.shape[1] - 400
    if new_ECG_data.shape[1] > 400:
        new_ECG_data_400 = new_ECG_data[:, :400]
        new_PGV_data_400 = new_PGV_data[:, :400]

    else:
        last_ECG_value_tensor = new_ECG_data[:, -1]
        last_ECG_value_tensor_view = last_ECG_value_tensor.view(
            last_ECG_value_tensor.shape[0], 1
        )
        last_ECG_value_view_tensors = torch.cat(
            [last_ECG_value_tensor_view] * (400 - new_ECG_data.shape[1]), dim=1
        )
        new_ECG_data_400 = torch.concat(
            [new_ECG_data, last_ECG_value_view_tensors], dim=1
        )

        last_PGV_value_tensor = new_PGV_data[:, -1]
        last_PGV_value_tensor_view = last_PGV_value_tensor.view(
            last_PGV_value_tensor.shape[0], 1
        )
        last_PGV_value_view_tensors = torch.cat(
            [last_PGV_value_tensor_view] * (400 - new_PGV_data.shape[1]), dim=1
        )
        new_PGV_data_400 = torch.concat(
            [new_PGV_data, last_PGV_value_view_tensors], dim=1
        )
    new_label_name = label_name + str(extation_rate)
    pt_array_augumentation = pt_array.copy()
    pt_array_augumentation[1] = pt_array_augumentation[1] + slide_index
    pt_array_augumentation[3] = pt_array_augumentation[3] + slide_index
    return new_ECG_data_400, new_PGV_data_400, new_label_name, pt_array_augumentation


def make_pq_extension_datas(PGV_datas, ECG_datas, pt_array, label_name, extation_rate):
    p_offset_org = pt_array[2]
    q_peak = pt_array[6]  # Correctly use q_peak for PQ interval end
    if p_offset_org >= q_peak:
        return ECG_datas, PGV_datas, label_name, pt_array

    extation_range_ECG = ECG_datas[:, p_offset_org:q_peak]
    extation_range_PGV = PGV_datas[:, p_offset_org:q_peak]

    if extation_range_ECG.shape[1] <= 0:
        return ECG_datas, PGV_datas, label_name, pt_array

    new_extation_range_ECG, new_extation_range_PGV = linear_interpolation_All(
        extation_range_ECG=extation_range_ECG,
        extation_range_PGV=extation_range_PGV,
        extation_rate=extation_rate,
    )
    new_ECG_data = torch.concat(
        [ECG_datas[:, :p_offset_org], new_extation_range_ECG, ECG_datas[:, q_peak:]],
        dim=1,
    )
    new_PGV_data = torch.concat(
        [PGV_datas[:, :p_offset_org], new_extation_range_PGV, PGV_datas[:, q_peak:]],
        dim=1,
    )
    slide_index = new_ECG_data.shape[1] - 400
    if new_ECG_data.shape[1] >= 400:
        new_ECG_data_400 = new_ECG_data[:, (new_ECG_data.shape[1] - 400) :]
        new_PGV_data_400 = new_PGV_data[:, (new_ECG_data.shape[1] - 400) :]
    else:
        First_ECG_value_tensor = new_ECG_data[:, 0]
        First_ECG_value_tensor_view = First_ECG_value_tensor.view(
            First_ECG_value_tensor.shape[0], 1
        )
        First_ECG_value_view_tensors = torch.cat(
            [First_ECG_value_tensor_view] * (400 - new_ECG_data.shape[1]), dim=1
        )
        new_ECG_data_400 = torch.concat(
            [First_ECG_value_view_tensors, new_ECG_data], dim=1
        )

        First_PGV_value_tensor = new_PGV_data[:, -1]
        First_PGV_value_tensor_view = First_PGV_value_tensor.view(
            First_PGV_value_tensor.shape[0], 1
        )
        First_PGV_value_view_tensors = torch.cat(
            [First_PGV_value_tensor_view] * (400 - new_PGV_data.shape[1]), dim=1
        )
        new_PGV_data_400 = torch.concat(
            [First_PGV_value_view_tensors, new_PGV_data], dim=1
        )

    new_label_name = label_name + "extraction_P=" + str(extation_rate)
    pt_array_augumentation = pt_array.copy()
    pt_array_augumentation[0] = pt_array_augumentation[0] - slide_index
    pt_array_augumentation[2] = pt_array_augumentation[2] - slide_index
    pt_array_augumentation[4] = pt_array_augumentation[4] - slide_index
    return new_ECG_data_400, new_PGV_data_400, new_label_name, pt_array_augumentation


def make_st_extension_datas(PGV_datas, ECG_datas, pt_array, label_name, extation_rate):
    t_onset_org = pt_array[3]
    s_peak = pt_array[7]  # Correctly use s_peak for ST segment start
    if s_peak >= t_onset_org:
        return ECG_datas, PGV_datas, label_name, pt_array

    extation_range_ECG = ECG_datas[:, s_peak:t_onset_org]
    extation_range_PGV = PGV_datas[:, s_peak:t_onset_org]

    if extation_range_ECG.shape[1] <= 0:
        return ECG_datas, PGV_datas, label_name, pt_array

    new_extation_range_ECG, new_extation_range_PGV = linear_interpolation_All(
        extation_range_ECG=extation_range_ECG,
        extation_range_PGV=extation_range_PGV,
        extation_rate=extation_rate,
    )
    new_ECG_data = torch.concat(
        [ECG_datas[:, :s_peak], new_extation_range_ECG, ECG_datas[:, t_onset_org:]],
        dim=1,
    )
    new_PGV_data = torch.concat(
        [PGV_datas[:, :s_peak], new_extation_range_PGV, PGV_datas[:, t_onset_org:]],
        dim=1,
    )
    slide_index = new_ECG_data.shape[1] - 400
    if new_ECG_data.shape[1] >= 400:
        new_ECG_data_400 = new_ECG_data[:, :400]
        new_PGV_data_400 = new_PGV_data[:, :400]
    else:
        last_ECG_value_tensor = new_ECG_data[:, -1]
        last_ECG_value_tensor_view = last_ECG_value_tensor.view(
            last_ECG_value_tensor.shape[0], 1
        )
        last_ECG_value_view_tensors = torch.cat(
            [last_ECG_value_tensor_view] * (400 - new_ECG_data.shape[1]), dim=1
        )
        new_ECG_data_400 = torch.concat(
            [new_ECG_data, last_ECG_value_view_tensors], dim=1
        )

        last_PGV_value_tensor = new_PGV_data[:, -1]
        last_PGV_value_tensor_view = last_PGV_value_tensor.view(
            last_PGV_value_tensor.shape[0], 1
        )
        last_PGV_value_view_tensors = torch.cat(
            [last_PGV_value_tensor_view] * (400 - new_PGV_data.shape[1]), dim=1
        )
        new_PGV_data_400 = torch.concat(
            [new_PGV_data, last_PGV_value_view_tensors], dim=1
        )
    new_label_name = label_name + str(extation_rate)
    pt_array_augumentation = pt_array.copy()
    pt_array_augumentation[1] = pt_array_augumentation[1] + slide_index
    pt_array_augumentation[3] = pt_array_augumentation[3] + slide_index
    pt_array_augumentation[7] = pt_array_augumentation[7] + slide_index
    return new_ECG_data_400, new_PGV_data_400, new_label_name, pt_array_augumentation


def sin_wave(point_num, extation_rate):
    A = extation_rate - 1
    frequency = 0.5
    if point_num <= 0:
        return 1.0  # Return a scalar float
    t = np.linspace(0, 1, int(point_num * 1), endpoint=True)
    y = A * np.sin(2 * np.pi * frequency * t) + 1
    return y


def make_p_height_extation(PGV_datas, ECG_datas, pt_array, label_name, extation_rate):
    p_onset_org = pt_array[0]
    p_offset_org = pt_array[2]

    if p_onset_org >= p_offset_org:
        return ECG_datas, PGV_datas, label_name, pt_array

    base_lines_tensor_ECG = ECG_datas[:, pt_array[0]]
    base_lines_tensor_PGV = PGV_datas[:, pt_array[0]]
    extation_range_ECG = ECG_datas[:, p_onset_org:p_offset_org]
    extation_range_PGV = PGV_datas[:, p_onset_org:p_offset_org]
    point_num = p_offset_org - p_onset_org

    if point_num <= 0:
        return ECG_datas, PGV_datas, label_name, pt_array

    extation_rate_sin = torch.tensor(sin_wave(point_num, extation_rate=extation_rate))
    extation_rate_sin = extation_rate_sin.float()
    new_extation_range_ECG = (
        extation_range_ECG - base_lines_tensor_ECG.view(ECG_datas.shape[0], 1)
    ) * extation_rate_sin + base_lines_tensor_ECG.view(ECG_datas.shape[0], 1)
    new_extation_range_PGV = (
        extation_range_PGV - base_lines_tensor_PGV.view(PGV_datas.shape[0], 1)
    ) * extation_rate_sin + base_lines_tensor_PGV.view(PGV_datas.shape[0], 1)
    new_ECG_data = torch.concat(
        [
            ECG_datas[:, :p_onset_org],
            new_extation_range_ECG,
            ECG_datas[:, p_offset_org:],
        ],
        dim=1,
    )
    new_PGV_data = torch.concat(
        [
            PGV_datas[:, :p_onset_org],
            new_extation_range_PGV,
            PGV_datas[:, p_offset_org:],
        ],
        dim=1,
    )
    new_label_name = label_name + str(extation_rate)
    return new_ECG_data, new_PGV_data, new_label_name, pt_array


def make_t_height_extation(PGV_datas, ECG_datas, pt_array, label_name, extation_rate):
    t_onset_org = pt_array[3]
    t_offset_org = pt_array[1]
    if t_onset_org >= t_offset_org:
        return ECG_datas, PGV_datas, label_name, pt_array

    base_lines_tensor_ECG = ECG_datas[:, pt_array[0]]
    base_lines_tensor_PGV = PGV_datas[:, pt_array[0]]
    extation_range_ECG = ECG_datas[:, t_onset_org:t_offset_org]
    extation_range_PGV = PGV_datas[:, t_onset_org:t_offset_org]
    point_num = t_offset_org - t_onset_org

    if point_num <= 0:
        return ECG_datas, PGV_datas, label_name, pt_array

    extation_rate_sin = torch.tensor(sin_wave(point_num, extation_rate=extation_rate))
    extation_rate_sin = extation_rate_sin.float()
    new_extation_range_ECG = (
        extation_range_ECG - base_lines_tensor_ECG.view(ECG_datas.shape[0], 1)
    ) * extation_rate_sin + base_lines_tensor_ECG.view(ECG_datas.shape[0], 1)
    new_extation_range_PGV = (
        extation_range_PGV - base_lines_tensor_PGV.view(PGV_datas.shape[0], 1)
    ) * extation_rate_sin + base_lines_tensor_PGV.view(PGV_datas.shape[0], 1)
    new_ECG_data = torch.concat(
        [
            ECG_datas[:, :t_onset_org],
            new_extation_range_ECG,
            ECG_datas[:, t_offset_org:],
        ],
        dim=1,
    )
    new_PGV_data = torch.concat(
        [
            PGV_datas[:, :t_onset_org],
            new_extation_range_PGV,
            PGV_datas[:, t_offset_org:],
        ],
        dim=1,
    )
    new_label_name = label_name + str(extation_rate)
    return new_ECG_data, new_PGV_data, new_label_name, pt_array


def create_dynamic_adj(x_batch):
    """
    Computes a batch of dynamic adjacency matrices using Pearson correlation.

    Args:
        x_batch (torch.Tensor): Input time-series data with shape
                                (batch_size, num_channels, sequence_length).

    Returns:
        torch.Tensor: Batch of adjacency matrices with shape
                      (batch_size, num_channels, num_channels).
                      Values are between -1 and 1.
    """
    batch_size, num_channels, sequence_length = x_batch.shape
    adj_batch = torch.zeros(
        batch_size, num_channels, num_channels, device=x_batch.device
    )

    for i in range(batch_size):
        # Extract a single sample (num_channels, sequence_length)
        sample = x_batch[i]

        # Calculate Pearson correlation matrix for this sample
        # Transpose to (sequence_length, num_channels) for torch.corrcoef
        # Or, manually calculate since torch.corrcoef only works for 1D for now if no custom impl
        # Let's use a manual calculation for clarity and direct control, similar to scipy's behavior

        # Mean center the data
        sample_mean = sample.mean(dim=1, keepdim=True)  # (num_channels, 1)
        sample_centered = sample - sample_mean  # (num_channels, sequence_length)

        # Calculate covariance matrix manually
        # cov(X, Y) = E[(X-mu_X)(Y-mu_Y)]
        # Sum of products for numerator
        numerator = torch.matmul(
            sample_centered, sample_centered.transpose(0, 1)
        )  # (num_channels, num_channels)

        # Denominator: product of standard deviations

        # Add a stabilizing epsilon to prevent division by zero or near-zero

        std_dev = torch.sqrt(torch.sum(sample_centered**2, dim=1, keepdim=True)) + 1e-8

        denominator = torch.matmul(std_dev, std_dev.transpose(0, 1))

        # Pearson correlation

        correlation_matrix = numerator / denominator

        adj_batch[i] = correlation_matrix

    return adj_batch
