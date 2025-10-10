# vae_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadVAE(nn.Module):
    """
    1つの共有エンコーダと、P波・R波・T波を再構成するための
    3つの専用デコーダ（ヘッド）を持つVAEモデル。
    """

    def __init__(
        self, input_dim, latent_dim, enc_hidden_dims=None, dec_hidden_dims=None
    ):
        super(MultiHeadVAE, self).__init__()
        enc_hidden_dims = enc_hidden_dims or [512, 256]
        self.encoder = nn.ModuleList()
        prev_dim = input_dim
        for h_dim in enc_hidden_dims:
            self.encoder.append(nn.Linear(prev_dim, h_dim))
            prev_dim = h_dim
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)

        dec_hidden_dims = dec_hidden_dims or [256, 512]
        self.decoder_p = self._build_decoder(latent_dim, dec_hidden_dims, input_dim)
        self.decoder_r = self._build_decoder(latent_dim, dec_hidden_dims, input_dim)
        self.decoder_t = self._build_decoder(latent_dim, dec_hidden_dims, input_dim)

    def _build_decoder(self, latent_dim, hidden_dims, output_dim):
        decoder = nn.ModuleList()
        prev_dim = latent_dim
        for h_dim in hidden_dims:
            decoder.append(nn.Linear(prev_dim, h_dim))
            prev_dim = h_dim
        decoder.append(nn.Linear(prev_dim, output_dim))
        return decoder

    def encode(self, x):
        h = x
        for layer in self.encoder:
            h = F.relu(layer(h))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, decoder_head):
        h = z
        for i, layer in enumerate(decoder_head):
            h = F.relu(layer(h)) if i < len(decoder_head) - 1 else layer(h)
        return h

    def forward(self, x):
        x_flat = x.view(x.size(0), -1)
        mu, logvar = self.encode(x_flat)
        z = self.reparameterize(mu, logvar)
        recon_x_p = self.decode(z, self.decoder_p).view(x.size())
        recon_x_r = self.decode(z, self.decoder_r).view(x.size())
        recon_x_t = self.decode(z, self.decoder_t).view(x.size())
        return recon_x_p, recon_x_r, recon_x_t, mu, logvar


def loss_function_multi_head(
    recon_p,
    recon_r,
    recon_t,
    target_x,
    mu,
    logvar,
    beta=1.0,
    weights={"P": 1.0, "R": 1.0, "T": 1.0},
    p_end=130,
    r_start=130,
    r_end=170,
    t_start=170,
):
    target_p = target_x.view(target_x.size(0), -1)[:, :p_end]
    target_r = target_x.view(target_x.size(0), -1)[:, r_start:r_end]
    target_t = target_x.view(target_x.size(0), -1)[:, t_start:]

    recon_p_segment = recon_p.view(recon_p.size(0), -1)[:, :p_end]
    recon_r_segment = recon_r.view(recon_r.size(0), -1)[:, r_start:r_end]
    recon_t_segment = recon_t.view(recon_t.size(0), -1)[:, t_start:]

    mse = nn.MSELoss(reduction="sum")
    mse_p = mse(recon_p_segment, target_p)
    mse_r = mse(recon_r_segment, target_r)
    mse_t = mse(recon_t_segment, target_t)

    reconstruction_loss = (
        weights["P"] * mse_p + weights["R"] * mse_r + weights["T"] * mse_t
    )
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    batch_size = target_x.size(0)
    total_loss = (reconstruction_loss + beta * KLD) / batch_size

    return total_loss, reconstruction_loss / batch_size, KLD / batch_size
