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


def main():
    utils.create_directory_if_not_exists("model_pth")
    args = config.get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    writer = SummaryWriter(
        log_dir=f"runs/{args.current_time}/{args.TARGET_NAME}/SimpleVAE_log"
    )

    train_dataset, test_dataset = Dataset.Dataset_setup_8ch_pt_augmentation(
        TARGET_NAME=args.TARGET_NAME,
        transform_type=args.transform_type,
        Dataset_name=args.Dataset_name,
        dataset_num=args.dataset_num,
        DataAugumentation=args.p_augumentation,  # Using p_aug as a generic one
        ave_data_flg=args.ave_data_flg,
        num_channels=args.num_channels,
    )

    ts = (
        datetime.datetime.fromtimestamp(time.time()).strftime("%Y%m%d_%H%M%S")
        + "_SimpleVAE"
        + "_TARGETNAME="
        + str(args.TARGET_NAME)
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
            utils.loss_fn_mse,  # Using standard MSE loss for simplicity
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

    # Test mode can be added here if needed


if __name__ == "__main__":
    main()
