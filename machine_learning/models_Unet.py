from audioop import bias
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import os


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHANNEL = 15


# class UNet1D(nn.Module):
#     def __init__(
#         self,
#         datalength,
#         enc_convlayer_sizes,
#         dec_convlayer_sizes,
#         in_channels=15,
#         out_channels=8,
#     ):
#         super(UNet1D, self).__init__()
#         self.datalength = datalength

#         self.encoders = nn.ModuleList()
#         self.decoders = nn.ModuleList()
#         self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
#         self.skip_connections = []

#         for out_channels, stride in enc_convlayer_sizes:
#             self.encoders.append(self.conv_block(in_channels, out_channels))
#             in_channels = out_channels

#         # Bottleneck
#         # self.bottleneck = self.conv_block(prev_channels, prev_channels * 2)
#         # prev_channels *= 2

#         # Expanding path
#         # self.decoders = nn.ModuleList()
#         # self.upconvs = nn.ModuleList()
#         for out_channels, stride in dec_convlayer_sizes:
#             self.decoders.append(
#                 nn.Sequential(
#                     nn.ConvTranspose1d(
#                         in_channels,
#                         out_channels,
#                         kernel_size=3,
#                         stride=stride,
#                         padding=1,
#                         output_padding=1 if stride > 1 else 0,
#                     ),
#                     nn.BatchNorm1d(out_channels),
#                     nn.ReLU(inplace=True),
#                 )
#             )
#             in_channels = out_channels

#         # 最終出力層
#         final_out_channels, final_stride = dec_convlayer_sizes[-1]
#         self.final_conv = nn.Conv1d(
#             in_channels,
#             out_channels,
#             kernel_size=1,
#         )

#     def conv_block(self, in_channels, out_channels):
#         return nn.Sequential(
#             nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
#             nn.BatchNorm1d(out_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
#             nn.BatchNorm1d(out_channels),
#             nn.ReLU(inplace=True),
#         )

#     def forward(self, x):

#         skips = []

#         # エンコーダ
#         for layer in self.encoders:
#             x = layer(x)
#             skips.append(x)

#         # デコーダ
#         for i, layer in enumerate(self.decoders):
#             print(x.shape)
#             print(layer(x).shape)
#             x = layer(x)

#             # スキップ接続（shapeが合う場合のみ）
#             if i < len(skips):
#                 skip = skips[-(i + 1)]
#                 if x.shape[-1] != skip.shape[-1]:
#                     x = F.pad(x, (0, skip.shape[-1] - x.shape[-1]))
#                 x = torch.cat((x, skip), dim=1)  # 加算ではなく結合（チャネル方向）

#         # 出力
#         x = self.final_conv(x)
#         return x


class UNet1D(nn.Module):
    def __init__(self, in_channels=15, out_channels=8, base_filters=32):
        super(UNet1D, self).__init__()

        # 畳み込み層3つ用
        # Encoder
        self.enc1 = self.encoder_block(in_channels, base_filters)
        self.enc2 = self.encoder_block(base_filters, base_filters * 2)
        self.enc3 = self.encoder_block(base_filters * 2, base_filters * 4)

        # Bridge
        self.bridge = self.conv_block(base_filters * 4, base_filters * 8)

        # Decoder
        self.dec1 = self.decoder_block(base_filters * 8, base_filters * 4)
        self.dec2 = self.decoder_block(base_filters * 4, base_filters * 2)
        self.dec3 = self.decoder_block(base_filters * 2, base_filters)

        # # Encoder
        # self.enc1 = self.encoder_block(in_channels, base_filters)
        # self.enc2 = self.encoder_block(base_filters, base_filters * 2)
        # self.enc3 = self.encoder_block(base_filters * 2, base_filters * 4)
        # self.enc4 = self.encoder_block(base_filters * 4, base_filters * 8)

        # # Bridge
        # self.bridge = self.conv_block(base_filters * 8, base_filters * 16)

        # # Decoder
        # self.dec1 = self.decoder_block(base_filters * 16, base_filters * 8)
        # self.dec2 = self.decoder_block(base_filters * 8, base_filters * 4)
        # self.dec3 = self.decoder_block(base_filters * 4, base_filters * 2)
        # self.dec4 = self.decoder_block(base_filters * 2, base_filters)

        # Output
        self.out_conv = nn.Conv1d(base_filters, out_channels, kernel_size=1)

        # MaxPool and Upsample
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def encoder_block(self, in_channels, out_channels):
        conv = self.conv_block(in_channels, out_channels)
        return conv

    def decoder_block(self, in_channels, out_channels):
        upconv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=2, stride=2)
        conv = self.conv_block(out_channels * 2, out_channels)
        return nn.ModuleDict({"upconv": upconv, "conv": conv})

    def forward(self, x):
        # 畳み込み層3つ用
        # Encoder
        x1 = self.enc1(x)
        p1 = self.pool(x1)
        x2 = self.enc2(p1)
        p2 = self.pool(x2)
        x3 = self.enc3(p2)
        p3 = self.pool(x3)
        # Bridge
        b = self.bridge(p3)
        # Decoder
        u1 = self.dec1["upconv"](b)
        u1 = self.pad_and_concat(u1, x3)
        u1 = self.dec1["conv"](u1)
        u2 = self.dec2["upconv"](u1)
        u2 = self.pad_and_concat(u2, x2)
        u2 = self.dec2["conv"](u2)
        u3 = self.dec3["upconv"](u2)
        u3 = self.pad_and_concat(u3, x1)
        u3 = self.dec3["conv"](u3)

        out = self.out_conv(u3)
        return out
        # # Encoder
        # x1 = self.enc1(x)
        # p1 = self.pool(x1)
        # x2 = self.enc2(p1)
        # p2 = self.pool(x2)
        # x3 = self.enc3(p2)
        # p3 = self.pool(x3)
        # x4 = self.enc4(p3)
        # p4 = self.pool(x4)

        # # Bridge
        # b = self.bridge(p4)

        # # Decoder
        # u1 = self.dec1["upconv"](b)
        # u1 = self.pad_and_concat(u1, x4)
        # u1 = self.dec1["conv"](u1)

        # u2 = self.dec2["upconv"](u1)
        # u2 = self.pad_and_concat(u2, x3)
        # u2 = self.dec2["conv"](u2)

        # u3 = self.dec3["upconv"](u2)
        # u3 = self.pad_and_concat(u3, x2)
        # u3 = self.dec3["conv"](u3)

        # u4 = self.dec4["upconv"](u3)
        # u4 = self.pad_and_concat(u4, x1)
        # u4 = self.dec4["conv"](u4)

        # out = self.out_conv(u4)
        # return out

    def pad_and_concat(self, upsampled, skip):
        """必要に応じてパディングしてチャネル方向に結合"""
        diff = skip.shape[-1] - upsampled.shape[-1]
        if diff > 0:
            upsampled = F.pad(upsampled, (0, diff))
        elif diff < 0:
            skip = F.pad(skip, (0, -diff))
        return torch.cat([upsampled, skip], dim=1)
