import numpy as np
import torch
import torch.nn as nn
import torch_geometric.nn as geo_nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.transforms import AddLaplacianEigenvectorPE

from torch_geometric.nn import GCNConv
from utils import create_dynamic_adj  # Import the new utility function


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def idx2onehot(idx, n):
    idx = idx.to(device)
    assert torch.max(idx).item() < n
    if idx.dim() == 1:
        idx = idx.unsqueeze(1)

    onehot = torch.zeros(idx.size(0), n).to(device)
    onehot.scatter_(1, idx, 1)

    return onehot


class VAE(nn.Module):
    def __init__(
        self,
        datalength,
        enc_convlayer_sizes,
        enc_fclayer_sizes,
        dec_fclayer_sizes,
        dec_convlayer_sizes,
        latent_size,
        conditional=False,
        num_labels=0,
        num_channels=16,  # New argument
    ):
        super().__init__()

        if conditional:
            assert num_labels > 0

        assert type(enc_convlayer_sizes) == list
        assert type(enc_fclayer_sizes) == list
        assert type(dec_fclayer_sizes) == list
        assert type(dec_convlayer_sizes) == list
        assert type(latent_size) == int

        self.latent_size = latent_size
        self.datalength = datalength

        self.encoder = Encoder(
            datalength,
            enc_convlayer_sizes,
            enc_fclayer_sizes,
            latent_size,
            conditional,
            num_labels,
            num_channels,  # Pass to Encoder
        ).to(device)
        self.decoder = Decoder(
            datalength,
            dec_convlayer_sizes,
            dec_fclayer_sizes,
            latent_size,
            conditional,
            num_labels,
        ).to(device)

    def forward(self, x, c=None):
        batch_size = x.size(0)
        means, log_var = self.encoder(x, c)
        std = torch.exp(0.5 * log_var).to(device)
        eps = torch.randn([batch_size, self.latent_size]).to(device)
        z = eps * std + means
        recon_x = self.decoder(z, c)
        return recon_x, means, log_var, z

    def inference(self, n=1, c=None):
        batch_size = n
        z = torch.randn([batch_size, self.latent_size]).to(device)
        recon_x = self.decoder(z, c)
        return recon_x


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(-1, input.size(1) * input.size(2))


class Encoder(nn.Module):
    def __init__(
        self,
        datalength,
        conv_layer_sizes,
        fc_layer_sizes,
        latent_size,
        conditional,
        num_labels,
        num_channels,  # New argument
    ):
        super().__init__()

        self.datalength = datalength
        self.conv_layer_sizes = conv_layer_sizes
        self.conditional = conditional
        self.num_channels = num_channels  # Store num_channels

        self.MLP_1 = nn.Sequential().to(device)
        self.MLP_2 = nn.Sequential().to(device)

        if len(conv_layer_sizes) != 0:
            for i, (conv_param_in, conv_param_out) in enumerate(
                (zip(conv_layer_sizes[:-1], conv_layer_sizes[1:]))
            ):
                self.MLP_1.add_module(
                    name=f"AC{i}",
                    module=nn.Conv1d(
                        conv_param_in[0],
                        conv_param_out[0],
                        kernel_size=6,
                        stride=conv_param_out[1],
                        padding=2,
                        bias=False,
                    ),
                )
                self.MLP_1.add_module(
                    name=f"AB{i}", module=nn.BatchNorm1d(conv_param_out[0])
                )
                self.MLP_1.add_module(name=f"AA{i}", module=nn.ReLU())
            self.MLP_1.add_module(name="F0", module=Flatten())

        for i, (in_size, out_size) in enumerate(
            zip(fc_layer_sizes[:-1], fc_layer_sizes[1:])
        ):
            self.MLP_2.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size)
            )
        self.MLP_2.add_module(name="A{:d}".format(i), module=nn.ReLU())

        self.linear_means = nn.Linear(fc_layer_sizes[-1], latent_size)
        self.linear_log_var = nn.Linear(fc_layer_sizes[-1], latent_size)

    def forward(self, x, c=None):
        if len(self.conv_layer_sizes) != 0:
            x = torch.reshape(
                x, (-1, self.num_channels, self.datalength)
            )  # Use self.num_channels

        if self.conditional:
            c = idx2onehot(c, n=10)
            x = torch.cat((x, c), dim=-1)

        x = self.MLP_1(x)
        x = self.MLP_2(x)

        means = self.linear_means(x).to(device)
        log_vars = self.linear_log_var(x).to(device)

        return means, log_vars


class Reshape(nn.Module):
    def __init__(self, re_channel, re_length):
        super().__init__()
        self.re_channel = re_channel
        self.re_length = re_length

    def forward(self, input):
        return torch.reshape(input, (-1, self.re_channel, self.re_length))


class Decoder(nn.Module):
    def __init__(
        self,
        datalength,
        conv_layer_sizes,
        fc_layer_sizes,
        latent_size,
        conditional,
        num_labels,
    ):
        super().__init__()
        self.datalength = datalength

        self.MLP = nn.Sequential().to(device)

        self.conditional = conditional
        if self.conditional:
            input_size = latent_size + num_labels
        else:
            input_size = latent_size

        for i, (in_size, out_size) in enumerate(
            zip([input_size] + fc_layer_sizes[:-1], fc_layer_sizes)
        ):
            self.MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size)
            )
            if i + 1 < len(fc_layer_sizes):
                self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())
            else:
                self.MLP.add_module(name="sigmoid", module=nn.Sigmoid())

        if len(conv_layer_sizes) != 0:
            self.MLP.add_module(
                name="R0",
                module=Reshape(
                    conv_layer_sizes[0][0],
                    int(fc_layer_sizes[-1] / conv_layer_sizes[0][0]),
                ),
            )
            for i, (conv_param_in, conv_param_out) in enumerate(
                (zip(conv_layer_sizes[:-1], conv_layer_sizes[1:]))
            ):
                self.MLP.add_module(
                    name=f"AC{i}",
                    module=nn.ConvTranspose1d(
                        conv_param_in[0],
                        conv_param_out[0],
                        kernel_size=6,
                        stride=int(conv_param_in[1]),
                        padding=2,
                        bias=False,
                    ),
                )

    def forward(self, z, c):
        if self.conditional:
            c = idx2onehot(c, n=10).to(device)
            z = torch.cat((z, c), dim=-1)

        x = self.MLP(z)
        x = torch.reshape(x, (-1, 1, self.datalength))
        return x


class DoubleConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


# class UNet1D(nn.Module):
#     def __init__(
#         self, num_channels=15, out_channels=8, base_filters=32
#     ):  # Changed in_channels to num_channels
#         super(UNet1D, self).__init__()
#         self.in_channels = num_channels  # Use num_channels
#         self.out_channels = out_channels

#         self.e1 = DoubleConv1d(self.in_channels, base_filters)  # Use self.in_channels
#         self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
#         self.drop1 = nn.Dropout(p=0.2)
#         self.e2 = DoubleConv1d(base_filters, base_filters * 2)
#         self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
#         self.drop2 = nn.Dropout(p=0.2)
#         self.e3 = DoubleConv1d(base_filters * 2, base_filters * 4)
#         self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
#         self.drop3 = nn.Dropout(p=0.2)
#         self.e4 = DoubleConv1d(base_filters * 4, base_filters * 8)
#         self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2)
#         self.drop4 = nn.Dropout(p=0.2)

#         self.bottleneck = DoubleConv1d(base_filters * 8, base_filters * 16)
#         self.drop_bottleneck = nn.Dropout(p=0.2)

#         self.up4 = nn.Sequential(
#             nn.Upsample(scale_factor=2, mode="linear", align_corners=True),
#             nn.Conv1d(base_filters * 16, base_filters * 8, kernel_size=3, padding=1),
#         )
#         self.d4 = DoubleConv1d(base_filters * 16, base_filters * 8)

#         self.up3 = nn.Sequential(
#             nn.Upsample(scale_factor=2, mode="linear", align_corners=True),
#             nn.Conv1d(base_filters * 8, base_filters * 4, kernel_size=3, padding=1),
#         )
#         self.d3 = DoubleConv1d(base_filters * 8, base_filters * 4)

#         self.up2 = nn.Sequential(
#             nn.Upsample(scale_factor=2, mode="linear", align_corners=True),
#             nn.Conv1d(base_filters * 4, base_filters * 2, kernel_size=3, padding=1),
#         )
#         self.d2 = DoubleConv1d(base_filters * 4, base_filters * 2)

#         self.up1 = nn.Sequential(
#             nn.Upsample(scale_factor=2, mode="linear", align_corners=True),
#             nn.Conv1d(base_filters * 2, base_filters, kernel_size=3, padding=1),
#         )
#         self.d1 = DoubleConv1d(base_filters * 2, base_filters)

#         self.out_conv = nn.Conv1d(base_filters, out_channels, kernel_size=1)

#     def forward(self, x):
#         s1 = self.e1(x)
#         p1 = self.drop1(self.pool1(s1))
#         s2 = self.e2(p1)
#         p2 = self.drop2(self.pool2(s2))
#         s3 = self.e3(p2)
#         p3 = self.drop3(self.pool3(s3))
#         s4 = self.e4(p3)
#         p4 = self.drop4(self.pool4(s4))

#         b = self.drop_bottleneck(self.bottleneck(p4))

#         u4 = self.up4(b)
#         if u4.shape[2] != s4.shape[2]:
#             u4 = F.interpolate(u4, size=s4.shape[2], mode="linear", align_corners=True)
#         c4 = torch.cat([u4, s4], dim=1)
#         d4 = self.d4(c4)

#         u3 = self.up3(d4)
#         if u3.shape[2] != s3.shape[2]:
#             u3 = F.interpolate(u3, size=s3.shape[2], mode="linear", align_corners=True)
#         c3 = torch.cat([u3, s3], dim=1)
#         d3 = self.d3(c3)

#         u2 = self.up2(d3)
#         if u2.shape[2] != s2.shape[2]:
#             u2 = F.interpolate(u2, size=s2.shape[2], mode="linear", align_corners=True)
#         c2 = torch.cat([u2, s2], dim=1)
#         d2 = self.d2(c2)

#         u1 = self.up1(d2)
#         if u1.shape[2] != s1.shape[2]:
#             u1 = F.interpolate(u1, size=s1.shape[2], mode="linear", align_corners=True)
#         c1 = torch.cat([u1, s1], dim=1)
#         d1 = self.d1(c1)

#         output = self.out_conv(d1)

#         return torch.sigmoid(output)


class UNet1D(nn.Module):
    def __init__(self, num_channels=15, out_channels=8, base_filters=32, depth=4):
        super(UNet1D, self).__init__()
        self.depth = depth

        # 各層を管理するリスト
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.drops = nn.ModuleList()  # 以前の drop1, drop2... に相当

        self.ups = nn.ModuleList()  # 以前の up4, up3... に相当
        self.decoders = nn.ModuleList()  # 以前の d4, d3... に相当

        # --- Encoder Path (Down-sampling) ---
        in_ch = num_channels
        for i in range(depth):
            out_ch = base_filters * (2**i)

            # DoubleConv (e1, e2...)
            self.encoders.append(DoubleConv1d(in_ch, out_ch))

            # MaxPool (pool1, pool2...)
            self.pools.append(nn.MaxPool1d(kernel_size=2, stride=2))

            # Dropout (drop1, drop2...) - 以前の構成を維持
            self.drops.append(nn.Dropout(p=0.2))

            in_ch = out_ch

        # --- Bottleneck ---
        # 以前の bottleneck と drop_bottleneck
        self.bottleneck = DoubleConv1d(in_ch, base_filters * (2**depth))
        self.drop_bottleneck = nn.Dropout(p=0.2)

        # --- Decoder Path (Up-sampling) ---
        # 深い層から浅い層へ向かって構築 (例: depth=4なら i=3,2,1,0)
        for i in range(depth - 1, -1, -1):
            in_ch_up = base_filters * (2 ** (i + 1))  # 下の層からの入力
            out_ch_up = base_filters * (2**i)  # Skip Connectionと同じサイズ

            # Upsample + Conv1d (以前の up4, up3... の構成)
            # TransposeConvではなく、Upsample(linear)を使用
            self.ups.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="linear", align_corners=True),
                    nn.Conv1d(in_ch_up, out_ch_up, kernel_size=3, padding=1),
                )
            )

            # DoubleConv (d4, d3...)
            # concatするので入力チャネルは2倍
            self.decoders.append(DoubleConv1d(out_ch_up * 2, out_ch_up))

        # --- Output Layer ---
        self.out_conv = nn.Conv1d(base_filters, out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # --- Encoder Forward ---
        for i in range(self.depth):
            x = self.encoders[i](x)
            skip_connections.append(x)  # Skip connection用に保存 (s1, s2...)

            x = self.pools[i](x)  # Pooling
            x = self.drops[i](x)  # Dropout (以前のモデル通りPoolingの後に適用)

        # --- Bottleneck Forward ---
        x = self.bottleneck(x)
        x = self.drop_bottleneck(x)  # Dropout

        # --- Decoder Forward ---
        # skip_connections は [s1, s2, s3, s4] の順なので、後ろから取り出す
        for i in range(self.depth):
            # Up-sampling (u4...)
            x = self.ups[i](x)

            # Skip connectionの取得 (s4...)
            skip = skip_connections.pop()

            # サイズ補正 (以前のモデルの `if u4.shape[2] != s4.shape[2]:` に相当)
            if x.shape[2] != skip.shape[2]:
                x = F.interpolate(
                    x, size=skip.shape[2], mode="linear", align_corners=True
                )

            # Concatenate (c4...)
            x = torch.cat([x, skip], dim=1)

            # DoubleConv (d4...)
            x = self.decoders[i](x)

        output = self.out_conv(x)

        # 0から1にクリッピング
        # return torch.clamp(output, 0.0, 1.0)
        return torch.sigmoid(output)


class SimpleCNN(nn.Module):
    def __init__(self, num_channels=15, out_channels=8, depth=4, init_filters=32):
        super(SimpleCNN, self).__init__()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.depth = depth

        # Encoder
        in_ch = num_channels
        for i in range(depth):
            out_ch = init_filters * (2**i)
            self.encoder.append(DoubleConv1d(in_ch, out_ch))
            self.encoder.append(nn.MaxPool1d(2))
            in_ch = out_ch

        # Bottleneck
        self.bottleneck = DoubleConv1d(in_ch, init_filters * (2**depth))

        # Decoder
        in_ch = init_filters * (2**depth)
        for i in reversed(range(depth)):
            out_ch = init_filters * (2**i)
            self.decoder.append(
                nn.ConvTranspose1d(in_ch, out_ch, kernel_size=2, stride=2)
            )
            self.decoder.append(DoubleConv1d(out_ch, out_ch))
            in_ch = out_ch

        self.out_conv = nn.Conv1d(in_ch, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder path
        for i in range(0, len(self.encoder), 2):
            x = self.encoder[i](x)
            x = self.encoder[i + 1](x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder path
        for i in range(0, len(self.decoder), 2):
            x = self.decoder[i](x)
            x = self.decoder[i + 1](x)

        return torch.sigmoid(self.out_conv(x))


class CNN_LSTM_Model(nn.Module):
    def __init__(
        self,
        num_channels=15,
        out_channels=8,
        cnn_filters_1=32,
        cnn_filters_2=64,
        cnn_kernel_size=5,
        lstm_hidden_size=128,
        lstm_num_layers=1,
    ):
        super(CNN_LSTM_Model, self).__init__()

        self.cnn_block = nn.Sequential(
            nn.Conv1d(
                in_channels=num_channels,
                out_channels=cnn_filters_1,
                kernel_size=cnn_kernel_size,
                padding="same",
            ),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=cnn_filters_1,
                out_channels=cnn_filters_2,
                kernel_size=cnn_kernel_size,
                padding="same",
            ),
            nn.ReLU(),
        )

        self.lstm = nn.LSTM(
            input_size=cnn_filters_2,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
        )

        self.fc = nn.Linear(in_features=lstm_hidden_size, out_features=out_channels)

    def forward(self, x):
        # Input shape: (batch_size, num_channels, datalength)

        # CNN processing
        x = self.cnn_block(x)
        # Shape after CNN: (batch_size, cnn_filters_2, datalength)

        # Prepare for LSTM
        x = x.permute(0, 2, 1)
        # Shape for LSTM: (batch_size, datalength, cnn_filters_2)

        # LSTM processing
        x, _ = self.lstm(x)
        # Shape after LSTM: (batch_size, datalength, lstm_hidden_size)

        # Fully connected layer processing
        x = self.fc(x)

        # Add Sigmoid activation to constrain output to [0, 1]
        x = torch.sigmoid(x)

        # Final permutation
        x = x.permute(0, 2, 1)
        # Final output shape: (batch_size, out_channels, datalength)

        return x


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, datalength):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.datalength = datalength
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(p=0.2)  # Added dropout layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: (batch_size, num_channels, datalength)
        # Permute to (batch_size, datalength, num_channels) for LSTM
        x = x.permute(0, 2, 1)  # (batch_size, datalength, input_size)

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        # out shape: (batch_size, datalength, hidden_size)
        out = self.fc(out)
        # out shape: (batch_size, datalength, output_size)

        # Add Sigmoid activation to constrain output to [0, 1]
        out = torch.sigmoid(out)

        # Permute back to (batch_size, output_size, datalength)
        out = out.permute(0, 2, 1)
        return out


# class ECGReconGNN(nn.Module):
#     def __init__(
#         self, input_dim, hidden_dim_rnn, hidden_dim_gcn, output_dim, sequence_length
#     ):
#         super(ECGReconGNN, self).__init__()
#         self.num_channels = input_dim  # 15 channels
#         self.output_channels = output_dim  # 8 channels
#         self.sequence_length = sequence_length

#         # 1. Temporal Encoder (Node Stream)
#         self.node_rnns = nn.ModuleList(
#             [
#                 nn.LSTM(input_size=1, hidden_size=hidden_dim_rnn, batch_first=True)
#                 for _ in range(self.num_channels)
#             ]
#         )

#         # 2. Spatial Mixer (GCN)
#         self.gcn = GCNConv(in_channels=hidden_dim_rnn, out_channels=hidden_dim_gcn)

#         # 3. Reconstruction Decoder
#         self.reconstruction_head = nn.Linear(
#             hidden_dim_gcn * self.num_channels, output_dim * sequence_length
#         )

#     def forward(self, x):
#         # x shape: (batch_size, num_channels, sequence_length)
#         if torch.isnan(x).any() or torch.isinf(x).any():
#             print("!!! DEBUG: NaN or Inf in input x")

#         batch_size, num_channels, _ = x.shape

#         # Step 1: Temporal Encoding for each node
#         node_features_list = []
#         for i in range(num_channels):
#             channel_data = x[:, i, :].unsqueeze(-1)

#             # Clamp the data to a reasonable range to prevent instability
#             channel_data = torch.clamp(channel_data, -5.0, 5.0)

#             # Pass through LSTM
#             _, (h_n, c_n) = self.node_rnns[i](channel_data)

#             # --- Enhanced Debugging ---
#             if torch.isnan(h_n).any() or torch.isinf(h_n).any():
#                 print(f"!!! CRITICAL: NaN/Inf detected in HIDDEN STATE for channel {i}")
#                 print(
#                     f"--- Input Stats for Failing Channel {i}: Min={channel_data.min().item():.4f}, Max={channel_data.max().item():.4f}, Mean={channel_data.mean().item():.4f}"
#                 )
#             if torch.isnan(c_n).any() or torch.isinf(c_n).any():
#                 print(f"!!! CRITICAL: NaN/Inf detected in CELL STATE for channel {i}")
#             # --- End Enhanced Debugging ---

#             node_features_list.append(h_n.squeeze(0))

#         # Concatenate node features: (batch_size, num_channels, hidden_dim_rnn)
#         node_features = torch.stack(node_features_list, dim=1)
#         if torch.isnan(node_features).any() or torch.isinf(node_features).any():
#             print("!!! DEBUG: NaN or Inf after Temporal Encoding (RNN)")

#         # Step 2: Dynamic Graph Construction (Adjacency Matrix)
#         adj_batch = create_dynamic_adj(x)
#         if torch.isnan(adj_batch).any() or torch.isinf(adj_batch).any():
#             print("!!! DEBUG: NaN or Inf in Adjacency Matrix")

#         # Step 3: Spatial Mixing (GCN)
#         gcn_outputs = []
#         for i in range(batch_size):
#             current_node_features = node_features[i]
#             current_adj = adj_batch[i]
#             edge_index = current_adj.nonzero(as_tuple=False).t().contiguous()
#             edge_weight = current_adj[edge_index[0], edge_index[1]]
#             gcn_out = self.gcn(current_node_features, edge_index, edge_weight)
#             gcn_outputs.append(gcn_out)

#         gcn_output_stacked = torch.stack(gcn_outputs, dim=0)
#         if (
#             torch.isnan(gcn_output_stacked).any()
#             or torch.isinf(gcn_output_stacked).any()
#         ):
#             print("!!! DEBUG: NaN or Inf after Spatial Mixing (GCN)")

#         # Step 4: Reconstruction Decoder
#         flattened_output = gcn_output_stacked.view(batch_size, -1)
#         reconstructed_flat = self.reconstruction_head(flattened_output)
#         reconstructed_ecg = reconstructed_flat.view(
#             batch_size, self.output_channels, self.sequence_length
#         )
#         if torch.isnan(reconstructed_ecg).any() or torch.isinf(reconstructed_ecg).any():
#             print("!!! DEBUG: NaN or Inf in final reconstructed_ecg")

#         return reconstructed_ecg


# class ECGReconGNN(nn.Module):
#     def __init__(
#         self, input_dim, hidden_dim_rnn, hidden_dim_gcn, output_dim, sequence_length
#     ):
#         super(ECGReconGNN, self).__init__()
#         self.num_channels = input_dim  # 15 channels
#         self.output_channels = output_dim  # 8 channels
#         self.sequence_length = sequence_length
#         self.hidden_dim_rnn = hidden_dim_rnn

#         # 1. Temporal Encoder (Node Stream) - LayerNormを追加
#         # ModuleListで個別に持つより、バッチ処理を工夫するか、安定化層を挟むのが定石です
#         self.node_rnns = nn.ModuleList(
#             [
#                 nn.LSTM(input_size=1, hidden_size=hidden_dim_rnn, batch_first=True)
#                 for _ in range(self.num_channels)
#             ]
#         )
#         # LSTM出力安定化のためのLayerNorm
#         self.ln_rnn = nn.LayerNorm(hidden_dim_rnn)

#         # 2. Spatial Mixer (GCN)
#         self.gcn = GCNConv(in_channels=hidden_dim_rnn, out_channels=hidden_dim_gcn)
#         self.act = nn.LeakyReLU(0.1)  # GCN後に活性化関数を追加推奨

#         # 3. Reconstruction Decoder
#         self.reconstruction_head = nn.Linear(
#             hidden_dim_gcn * self.num_channels, output_dim * sequence_length
#         )

#         self._init_weights()

#     def _init_weights(self):
#         # LSTMの重みを直交行列で初期化（長期依存性の学習安定化）
#         for lstm in self.node_rnns:
#             for name, param in lstm.named_parameters():
#                 if "weight_ih" in name:
#                     nn.init.xavier_uniform_(param.data)
#                 elif "weight_hh" in name:
#                     nn.init.orthogonal_(param.data)
#                 elif "bias" in name:
#                     param.data.fill_(0)

#     def forward(self, x):
#         # x shape: (batch_size, num_channels, sequence_length)
#         batch_size, num_channels, seq_len = x.shape

#         # Step 1: Temporal Encoding
#         node_features_list = []
#         for i in range(num_channels):
#             # チャネルごとのデータを抽出 (B, L, 1)
#             channel_data = x[:, i, :].unsqueeze(-1)

#             # --- 修正点: 入力値の安全性チェック ---
#             # NaNがあれば0置換するなど、強制的な防御を入れることも検討
#             if torch.isnan(channel_data).any():
#                 channel_data = torch.nan_to_num(channel_data, nan=0.0)

#             # LSTM Forward
#             # self.node_rnns[i].flatten_parameters() # GPUメモリ効率化のおまじない
#             _, (h_n, c_n) = self.node_rnns[i](channel_data)

#             # --- 修正点: NaN発生時のガード (推論時用、学習時はlossで弾く) ---
#             if torch.isnan(h_n).any():
#                 # ここでログを出してもすでに遅い（重みが壊れている）ことが多いですが念のため
#                 print(f"Warning: NaN in channel {i}, resetting to zeros.")
#                 h_n = torch.zeros_like(h_n)

#             node_features_list.append(h_n.squeeze(0))

#         # (batch_size, num_channels, hidden_dim_rnn)
#         node_features = torch.stack(node_features_list, dim=1)

#         # --- 修正点: LayerNormの適用 ---
#         # 時間方向の特徴量のスケールを整える
#         node_features = self.ln_rnn(node_features)

#         # Step 2: Dynamic Graph Construction
#         # (create_dynamic_adjの実装によりますが、ここでのNaNチェックも重要)
#         adj_batch = create_dynamic_adj(x)

#         # Step 3: Spatial Mixing (GCN)
#         gcn_outputs = []
#         for i in range(batch_size):
#             current_node_features = node_features[i]
#             current_adj = adj_batch[i]
#             edge_index = current_adj.nonzero(as_tuple=False).t().contiguous()

#             # エッジがない場合の例外処理
#             if edge_index.numel() == 0:
#                 gcn_out = torch.zeros(
#                     self.num_channels, self.gcn.out_channels, device=x.device
#                 )
#             else:
#                 edge_weight = current_adj[edge_index[0], edge_index[1]]
#                 gcn_out = self.gcn(current_node_features, edge_index, edge_weight)
#                 gcn_out = self.act(gcn_out)  # 活性化関数

#             gcn_outputs.append(gcn_out)

#         gcn_output_stacked = torch.stack(gcn_outputs, dim=0)

#         # Step 4: Reconstruction Decoder
#         flattened_output = gcn_output_stacked.view(batch_size, -1)
#         reconstructed_flat = self.reconstruction_head(flattened_output)

#         reconstructed_ecg = reconstructed_flat.view(
#             batch_size, self.output_channels, self.sequence_length
#         )
#         return reconstructed_ecg


# --- GNN Block (Reference準拠) ---
class GNNx2(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNNx2, self).__init__()
        self.gnn1 = geo_nn.SSGConv(input_dim, hidden_dim, alpha=0.05)
        self.gnn2 = geo_nn.SSGConv(hidden_dim, output_dim, alpha=0.05)
        self.activate = nn.Tanh()  # 安定化のためTanh

        if input_dim != output_dim:
            self.skip = nn.Linear(input_dim, output_dim)
        else:
            self.skip = nn.Identity()

    def forward(self, x, edge_index, edge_weight):
        residual = self.skip(x)
        x = self.gnn1(x, edge_index, edge_weight)
        x = self.activate(x)
        x = self.gnn2(x, edge_index, edge_weight)
        return x + residual


# --- Main Model ---
class ECGReconGNN(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim_rnn,
        hidden_dim_gcn,
        output_dim,
        sequence_length,
        rnn_type="lstm",  # 'lstm' または 'gru' を指定可能
    ):
        super(ECGReconGNN, self).__init__()

        self.num_channels = input_dim
        self.hidden_dim_rnn = hidden_dim_rnn
        self.hidden_dim_gcn = hidden_dim_gcn
        self.output_channels = output_dim
        self.sequence_length = sequence_length
        self.rnn_type = rnn_type.lower()

        self.pe_dim = 8  # Laplacian PE dimension

        # 1. Input Stabilization (NaN対策)
        self.input_act = nn.Tanh()

        # 2. Temporal Encoder (LSTM / GRU)
        # Mambaを廃止し、LSTM/GRUの切り替え式に変更
        if self.rnn_type == "lstm":
            self.temporal = nn.LSTM(
                input_size=1, hidden_size=hidden_dim_rnn, batch_first=True
            )
            print(">> Using LSTM for Temporal Encoding")
        elif self.rnn_type == "gru":
            self.temporal = nn.GRU(
                input_size=1, hidden_size=hidden_dim_rnn, batch_first=True
            )
            print(">> Using GRU for Temporal Encoding")
        else:
            raise ValueError("rnn_type must be 'lstm' or 'gru'")

        self.ln_rnn = nn.LayerNorm(hidden_dim_rnn)

        # 3. Laplacian Positional Encoding
        self.laplacian_pe = AddLaplacianEigenvectorPE(k=self.pe_dim)

        # 4. Spatial Mixer (GNN)
        gnn_in_dim = hidden_dim_rnn + self.pe_dim

        # エッジ重み生成用
        self.edge_net = nn.Sequential(
            nn.Linear(hidden_dim_rnn, 1), nn.Softplus()  # 重みは正の値
        )

        self.gnn = GNNx2(gnn_in_dim, hidden_dim_gcn, hidden_dim_gcn)

        # 5. Reconstruction Decoder
        self.reconstruction_head = nn.Sequential(
            nn.Linear(hidden_dim_gcn * self.num_channels, 256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),
            nn.Linear(256, self.output_channels * self.sequence_length),
        )

    def forward(self, x):
        # x shape: (batch_size, num_channels, sequence_length)
        batch_size, num_channels, seq_len = x.shape
        device = x.device

        # --- Step 0: Input Guard ---
        x = self.input_act(x)  # NaN防止

        # --- Step 1: Temporal Encoding ---
        # (B, C, L) -> (B*C, L, 1)
        x_reshaped = x.view(batch_size * num_channels, seq_len, 1)

        # LSTMとGRUで戻り値の形式が異なるため分岐処理
        if self.rnn_type == "lstm":
            _, (h_n, _) = self.temporal(x_reshaped)
        else:  # gru
            _, h_n = self.temporal(x_reshaped)

        # h_n shape: (num_layers, batch*channels, hidden_dim)
        # 最終層の隠れ状態を取得 -> (B*C, Hidden)
        node_features = h_n[-1]

        # LayerNorm & Reshape -> (B, C, Hidden)
        node_features = self.ln_rnn(node_features)
        node_features = node_features.view(batch_size, num_channels, -1)

        # --- Step 2: Graph Construction & GNN ---
        gnn_outputs = []

        # 完全結合エッジテンプレート
        node_indices = torch.arange(num_channels, device=device)
        edge_index_template = torch.stack(
            torch.meshgrid(node_indices, node_indices, indexing="ij")
        ).reshape(2, -1)

        # PE計算 (構造情報)
        pe_data = Data(edge_index=edge_index_template, num_nodes=num_channels)
        pe_data = self.laplacian_pe(pe_data)
        pe = pe_data.laplacian_eigenvector_pe.to(device)  # (C, pe_dim)

        for i in range(batch_size):
            curr_nodes = node_features[i]  # (C, Hidden)

            # 特徴量 + PE
            curr_nodes_with_pe = torch.cat([curr_nodes, pe], dim=-1)

            # 動的エッジ重み (簡易Attention)
            weights = self.edge_net(curr_nodes).view(-1)
            edge_weights = weights[edge_index_template[0]]

            # NaN防止: 正規化
            edge_weights = edge_weights / (edge_weights.max() + 1e-6)

            # GNN Forward
            out = self.gnn(curr_nodes_with_pe, edge_index_template, edge_weights)
            gnn_outputs.append(out.flatten())

        # (B, C * GCN_Hidden)
        gnn_output_stacked = torch.stack(gnn_outputs, dim=0)

        # --- Step 3: Reconstruction ---
        flat_out = self.reconstruction_head(gnn_output_stacked)
        reconstructed_ecg = flat_out.view(
            batch_size, self.output_channels, self.sequence_length
        )

        return reconstructed_ecg
