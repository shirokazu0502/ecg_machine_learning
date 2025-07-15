from audioop import bias
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import os


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHANNEL = 15


# # --- 1. 新しい中核ブロック: Residual Inception Block ---
# # アーキテクチャ図の緑色のブロックを実装します。
# class ResidualInceptionBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
#         super(ResidualInceptionBlock, self).__init__()

#         # 中間チャネル数を計算 (4つのパスに分岐するため)
#         inter_channels = out_channels // 4

#         # --- 4つの並列パス ---
#         # Path 1: Conv
#         self.path1 = nn.Sequential(
#             nn.Conv1d(in_channels, inter_channels, kernel_size=1, padding="same"),
#             nn.BatchNorm1d(inter_channels),
#             nn.ReLU(inplace=True),
#         )

#         # Path 2: Conv -> Conv
#         self.path2 = nn.Sequential(
#             nn.Conv1d(in_channels, inter_channels, kernel_size=1, padding="same"),
#             nn.BatchNorm1d(inter_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv1d(
#                 inter_channels, inter_channels, kernel_size=kernel_size, padding="same"
#             ),
#             nn.BatchNorm1d(inter_channels),
#             nn.ReLU(inplace=True),
#         )

#         # Path 3: Conv -> Dilated Conv
#         self.path3 = nn.Sequential(
#             nn.Conv1d(in_channels, inter_channels, kernel_size=1, padding="same"),
#             nn.BatchNorm1d(inter_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv1d(
#                 inter_channels,
#                 inter_channels,
#                 kernel_size=kernel_size,
#                 padding=dilation,
#                 dilation=dilation,
#             ),
#             nn.BatchNorm1d(inter_channels),
#             nn.ReLU(inplace=True),
#         )

#         # Path 4: MaxPool -> Conv
#         self.path4 = nn.Sequential(
#             nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
#             nn.Conv1d(in_channels, inter_channels, kernel_size=1, padding="same"),
#             nn.BatchNorm1d(inter_channels),
#             nn.ReLU(inplace=True),
#         )

#         # --- パスの結合後 ---
#         # 結合後の畳み込み層
#         self.conv_linear = nn.Sequential(
#             nn.Conv1d(inter_channels * 4, out_channels, kernel_size=1, padding="same"),
#             nn.BatchNorm1d(out_channels),
#             nn.ReLU(inplace=True),
#         )

#         # --- 残差接続 (Residual Connection) ---
#         # 入力と出力のチャネル数が違う場合、チャネル数を合わせるための1x1 Conv
#         if in_channels != out_channels:
#             self.shortcut = nn.Conv1d(
#                 in_channels, out_channels, kernel_size=1, padding="same"
#             )
#         else:
#             self.shortcut = nn.Identity()  # チャネル数が同じ場合は何もしない

#     def forward(self, x):
#         # 4つのパスの出力を計算
#         out1 = self.path1(x)
#         out2 = self.path2(x)
#         out3 = self.path3(x)
#         out4 = self.path4(x)

#         # パスの出力をチャネル方向に結合
#         x_concat = torch.cat([out1, out2, out3, out4], dim=1)

#         # 結合後の処理
#         x_linear = self.conv_linear(x_concat)

#         # 残差接続 (入力xを足し合わせる)
#         shortcut_out = self.shortcut(x)
#         out = x_linear + shortcut_out
#         return F.relu(out)


# # --- 2. モデル全体: Residual Inception U-Net ---
# # 新しいブロックを使ってU-Net全体を構築します。
# class UNet1D(nn.Module):
#     def __init__(self, in_channels=15, out_channels=8, base_filters=32):
#         super(UNet1D, self).__init__()

#         # --- Encoder ---
#         self.enc1 = ResidualInceptionBlock(in_channels, base_filters)
#         self.enc2 = ResidualInceptionBlock(base_filters, base_filters * 2)
#         self.enc3 = ResidualInceptionBlock(base_filters * 2, base_filters * 4)

#         self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

#         # --- Bridge ---
#         self.bridge = ResidualInceptionBlock(base_filters * 4, base_filters * 8)

#         # --- Decoder ---
#         self.up1 = nn.ConvTranspose1d(
#             base_filters * 8, base_filters * 4, kernel_size=2, stride=2
#         )
#         self.dec1 = ResidualInceptionBlock(
#             base_filters * 8, base_filters * 4
#         )  # skip接続と結合するため入力は*2

#         self.up2 = nn.ConvTranspose1d(
#             base_filters * 4, base_filters * 2, kernel_size=2, stride=2
#         )
#         self.dec2 = ResidualInceptionBlock(base_filters * 4, base_filters * 2)

#         self.up3 = nn.ConvTranspose1d(
#             base_filters * 2, base_filters, kernel_size=2, stride=2
#         )
#         self.dec3 = ResidualInceptionBlock(base_filters * 2, base_filters)

#         # --- Output Layer ---
#         self.out_conv = nn.Conv1d(base_filters, out_channels, kernel_size=1)

#     def forward(self, x):
#         # --- Encoder ---
#         s1 = self.enc1(x)
#         p1 = self.pool(s1)

#         s2 = self.enc2(p1)
#         p2 = self.pool(s2)

#         s3 = self.enc3(p2)
#         p3 = self.pool(s3)

#         # --- Bridge ---
#         b = self.bridge(p3)

#         # --- Decoder ---
#         u1 = self.up1(b)
#         # スキップ接続 (エンコーダの出力s3と結合)
#         c1 = torch.cat([u1, s3], dim=1)
#         d1 = self.dec1(c1)

#         u2 = self.up2(d1)
#         # スキップ接続 (エンコーダの出力s2と結合)
#         c2 = torch.cat([u2, s2], dim=1)
#         d2 = self.dec2(c2)

#         u3 = self.up3(d2)
#         # スキップ接続 (エンコーダの出力s1と結合)
#         c3 = torch.cat([u3, s1], dim=1)
#         d3 = self.dec3(c3)

#         # --- Output ---
#         output = self.out_conv(d3)

#         return output

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv1d(nn.Module):
    """(convolution => [BN] => ReLU) * 2 のブロック"""

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


class UNet1D(nn.Module):
    """
    1次元データのためのU-Netアーキテクチャ

    Args:
        in_channels (int): 入力テンソルのチャンネル数 (次元数)
        out_channels (int): 出力テンソルのチャンネル数 (次元数)
    """

    def __init__(self, in_channels=15, out_channels=8, base_filters=32):
        super(UNet1D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # --- エンコーダ (ダウンサンプリング) ---
        self.e1 = DoubleConv1d(in_channels, 64)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.e2 = DoubleConv1d(64, 128)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.e3 = DoubleConv1d(128, 256)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)

        # --- ボトルネック ---
        self.bottleneck = DoubleConv1d(256, 512)

        # --- デコーダ (アップサンプリング) ---
        # チャンネル数を半分にしながら、シーケンス長を2倍にする
        self.up3 = nn.ConvTranspose1d(512, 256, kernel_size=2, stride=2)
        # スキップ接続と結合するため、入力チャンネル数は (256+256)=512
        self.d3 = DoubleConv1d(512, 256)

        self.up2 = nn.ConvTranspose1d(256, 128, kernel_size=2, stride=2)
        # 入力チャンネル数は (128+128)=256
        self.d2 = DoubleConv1d(256, 128)

        self.up1 = nn.ConvTranspose1d(128, 64, kernel_size=2, stride=2)
        # 入力チャンネル数は (64+64)=128
        self.d1 = DoubleConv1d(128, 64)

        # --- 出力層 ---
        self.out_conv = nn.Conv1d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # --- エンコーダ ---
        s1 = self.e1(x)  # スキップ接続用に保存
        p1 = self.pool1(s1)
        s2 = self.e2(p1)  # スキップ接続用に保存
        p2 = self.pool2(s2)
        s3 = self.e3(p2)  # スキップ接続用に保存
        p3 = self.pool3(s3)

        # --- ボトルネック ---
        b = self.bottleneck(p3)

        # --- デコーダ ---
        # ステージ3
        u3 = self.up3(b)
        # プーリングによりシーケンス長が奇数→偶数になった場合、サイズが1ずれることがあるため補間
        if u3.shape[2] != s3.shape[2]:
            u3 = F.interpolate(u3, size=s3.shape[2], mode="linear", align_corners=True)
        c3 = torch.cat([u3, s3], dim=1)  # チャンネル次元で結合
        d3 = self.d3(c3)

        # ステージ2
        u2 = self.up2(d3)
        if u2.shape[2] != s2.shape[2]:
            u2 = F.interpolate(u2, size=s2.shape[2], mode="linear", align_corners=True)
        c2 = torch.cat([u2, s2], dim=1)
        d2 = self.d2(c2)

        # ステージ1
        u1 = self.up1(d2)
        if u1.shape[2] != s1.shape[2]:
            u1 = F.interpolate(u1, size=s1.shape[2], mode="linear", align_corners=True)
        c1 = torch.cat([u1, s1], dim=1)
        d1 = self.d1(c1)

        # --- 出力 ---
        output = self.out_conv(d1)

        # 最終的な活性化関数 (Softmax, Sigmoidなど) は、
        # 損失関数の設計に応じてモデルの外で適用するのが一般的です。
        # 例えば nn.CrossEntropyLoss を使う場合は、このままの出力 (logits) を渡します。

        return output
