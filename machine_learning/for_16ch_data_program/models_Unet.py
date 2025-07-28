from audioop import bias
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import os

torch.autograd.set_detect_anomaly(True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHANNEL = 16


# 基本ブロックDoubleConv1dは変更なし
class DoubleConv1d(nn.Module):
    """(convolution => [BN] => ReLU) * 2 のブロック"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv1d(
                in_channels,
                mid_channels,
                kernel_size=3,
                padding=1,
                padding_mode="reflect",
                bias=False,
            ),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Conv1d(
                mid_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                padding_mode="reflect",
                bias=False,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class UNet1D(nn.Module):
    """
    5階層の1次元U-Netアーキテクチャ
    """

    def __init__(self, in_channels=16, out_channels=8, base_filters=64):
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
        self.e4 = DoubleConv1d(256, 512)
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.e5 = DoubleConv1d(512, 1024)  # 追加
        self.pool5 = nn.MaxPool1d(kernel_size=2, stride=2)  # 追加

        # --- ボトルネック (最深部) ---
        self.bottleneck = DoubleConv1d(1024, 2048)

        # --- デコーダ (アップサンプリング) ---
        self.up5 = nn.ConvTranspose1d(2048, 1024, kernel_size=2, stride=2)  # 追加
        self.d5 = DoubleConv1d(2048, 1024)  # 追加

        self.up4 = nn.ConvTranspose1d(1024, 512, kernel_size=2, stride=2)
        self.d4 = DoubleConv1d(1024, 512)

        self.up3 = nn.ConvTranspose1d(512, 256, kernel_size=2, stride=2)
        self.d3 = DoubleConv1d(512, 256)

        self.up2 = nn.ConvTranspose1d(256, 128, kernel_size=2, stride=2)
        self.d2 = DoubleConv1d(256, 128)

        self.up1 = nn.ConvTranspose1d(128, 64, kernel_size=2, stride=2)
        self.d1 = DoubleConv1d(128, 64)

        # --- 出力層 ---
        self.out_conv = nn.Conv1d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # --- エンコーダ ---
        s1 = self.e1(x)
        p1 = self.pool1(s1)
        s2 = self.e2(p1)
        p2 = self.pool2(s2)
        s3 = self.e3(p2)
        p3 = self.pool3(s3)
        s4 = self.e4(p3)
        p4 = self.pool4(s4)
        s5 = self.e5(p4)  # 追加
        p5 = self.pool5(s5)  # 追加

        # --- ボトルネック ---
        b = self.bottleneck(p5)

        # --- デコーダ ---
        # ステージ5 (追加)
        u5 = self.up5(b)
        if u5.shape[2] != s5.shape[2]:
            u5 = F.interpolate(u5, size=s5.shape[2], mode="linear", align_corners=True)
        c5 = torch.cat([u5, s5], dim=1)
        d5 = self.d5(c5)

        # ステージ4
        u4 = self.up4(d5)
        if u4.shape[2] != s4.shape[2]:
            u4 = F.interpolate(u4, size=s4.shape[2], mode="linear", align_corners=True)
        c4 = torch.cat([u4, s4], dim=1)
        d4 = self.d4(c4)

        # ステージ3
        u3 = self.up3(d4)
        if u3.shape[2] != s3.shape[2]:
            u3 = F.interpolate(u3, size=s3.shape[2], mode="linear", align_corners=True)
        c3 = torch.cat([u3, s3], dim=1)
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

        return output


# --------------------------
# Residual Block for 1D Conv
# --------------------------
# class ResBlock1d(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
#         super(ResBlock1d, self).__init__()
#         padding = kernel_size // 2
#         self.conv1 = nn.Conv1d(
#             in_channels, out_channels, kernel_size, stride, padding, bias=False
#         )
#         self.bn1 = nn.BatchNorm1d(out_channels)
#         self.relu = nn.LeakyReLU(0.2, inplace=True)
#         self.conv2 = nn.Conv1d(
#             out_channels, out_channels, kernel_size, stride, padding, bias=False
#         )
#         self.bn2 = nn.BatchNorm1d(out_channels)

#         # Skip connection
#         self.skip = nn.Conv1d(
#             in_channels, out_channels, kernel_size=1, stride=1, bias=False
#         )

#     def forward(self, x):
#         identity = self.skip(x)
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#         out = self.conv2(out)
#         out = self.bn2(out)
#         out += identity
#         out = self.relu(out)
#         return out


# class UNet1D(nn.Module):
#     def __init__(self, in_channels=16, out_channels=8):
#         super(UNet1D, self).__init__()

#         # Encoder
#         self.e1 = ResBlock1d(in_channels, 64)
#         self.pool1 = nn.MaxPool1d(2)
#         self.e2 = ResBlock1d(64, 128)
#         self.pool2 = nn.MaxPool1d(2)
#         self.e3 = ResBlock1d(128, 256)
#         self.pool3 = nn.MaxPool1d(2)
#         self.e4 = ResBlock1d(256, 512)
#         self.pool4 = nn.MaxPool1d(2)
#         self.e5 = ResBlock1d(512, 1024)
#         self.pool5 = nn.MaxPool1d(2)

#         # Bottleneck
#         self.bottleneck = ResBlock1d(1024, 2048)

#         # Decoder
#         self.up5 = nn.Sequential(
#             nn.ConvTranspose1d(2048, 1024, kernel_size=2, stride=2),
#             nn.BatchNorm1d(1024),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.5),
#         )
#         self.d5 = ResBlock1d(2048, 1024)

#         self.up4 = nn.Sequential(
#             nn.ConvTranspose1d(1024, 512, kernel_size=2, stride=2),
#             nn.BatchNorm1d(512),
#             nn.ReLU(inplace=True),
#         )
#         self.d4 = ResBlock1d(1024, 512)

#         self.up3 = nn.Sequential(
#             nn.ConvTranspose1d(512, 256, kernel_size=2, stride=2),
#             nn.BatchNorm1d(256),
#             nn.ReLU(inplace=True),
#         )
#         self.d3 = ResBlock1d(512, 256)

#         self.up2 = nn.Sequential(
#             nn.ConvTranspose1d(256, 128, kernel_size=2, stride=2),
#             nn.BatchNorm1d(128),
#             nn.ReLU(inplace=True),
#         )
#         self.d2 = ResBlock1d(256, 128)

#         self.up1 = nn.Sequential(
#             nn.ConvTranspose1d(128, 64, kernel_size=2, stride=2),
#             nn.BatchNorm1d(64),
#             nn.ReLU(inplace=True),
#         )
#         self.d1 = ResBlock1d(128, 64)

#         # Output
#         self.out_conv = nn.Conv1d(64, out_channels, kernel_size=1)
#         self.activation = nn.Sigmoid()

#     def forward(self, x):
#         # Encoder
#         s1 = self.e1(x)
#         p1 = self.pool1(s1)
#         s2 = self.e2(p1)
#         p2 = self.pool2(s2)
#         s3 = self.e3(p2)
#         p3 = self.pool3(s3)
#         s4 = self.e4(p3)
#         p4 = self.pool4(s4)
#         s5 = self.e5(p4)
#         p5 = self.pool5(s5)

#         # Bottleneck
#         b = self.bottleneck(p5)

#         # Decoder
#         u5 = self.up5(b)
#         if u5.shape[2] != s5.shape[2]:
#             u5 = F.interpolate(u5, size=s5.shape[2], mode="linear", align_corners=True)
#         d5 = self.d5(torch.cat([u5, s5], dim=1))

#         u4 = self.up4(d5)
#         if u4.shape[2] != s4.shape[2]:
#             u4 = F.interpolate(u4, size=s4.shape[2], mode="linear", align_corners=True)
#         d4 = self.d4(torch.cat([u4, s4], dim=1))

#         u3 = self.up3(d4)
#         if u3.shape[2] != s3.shape[2]:
#             u3 = F.interpolate(u3, size=s3.shape[2], mode="linear", align_corners=True)
#         d3 = self.d3(torch.cat([u3, s3], dim=1))

#         u2 = self.up2(d3)
#         if u2.shape[2] != s2.shape[2]:
#             u2 = F.interpolate(u2, size=s2.shape[2], mode="linear", align_corners=True)
#         d2 = self.d2(torch.cat([u2, s2], dim=1))

#         u1 = self.up1(d2)
#         if u1.shape[2] != s1.shape[2]:
#             u1 = F.interpolate(u1, size=s1.shape[2], mode="linear", align_corners=True)
#         d1 = self.d1(torch.cat([u1, s1], dim=1))

#         out = self.out_conv(d1)
#         return self.activation(out)
