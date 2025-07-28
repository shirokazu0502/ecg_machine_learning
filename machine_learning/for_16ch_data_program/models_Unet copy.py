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


# class DoubleConv(nn.Module):
#     """(Convolution => [BN] => ReLU) * 2 + Dropout"""

#     def __init__(self, in_channels, out_channels, mid_channels=None, dropout=False):
#         super().__init__()
#         if not mid_channels:
#             mid_channels = out_channels

#         layers = [
#             nn.Conv1d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm1d(mid_channels),
#             nn.ReLU(inplace=False),  # inplaceをTrueに変更してメモリ効率を改善
#             nn.Conv1d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm1d(out_channels),
#             nn.ReLU(inplace=False),
#         ]

#         if dropout:
#             # [cite_start]論文の記述に基づきDropoutを追加 [cite: 178]
#             layers.append(nn.Dropout(0.5))

#         self.double_conv = nn.Sequential(*layers)

#     def forward(self, x):
#         return self.double_conv(x)


# class UNet1D(nn.Module):
#     """
#     論文のモデルを5階層に拡張したUNet-BiLSTMモデル
#     """

#     def __init__(self, in_channels=1, out_channels=1):
#         super(UNet1D, self).__init__()

#         # --- エンコーダ (5階層) ---
#         self.enc1 = DoubleConv(in_channels, 128, dropout=False)
#         self.down1 = nn.Conv1d(128, 128, kernel_size=4, stride=2, padding=1)

#         self.enc2 = DoubleConv(128, 256, dropout=True)
#         self.down2 = nn.Conv1d(256, 256, kernel_size=4, stride=2, padding=1)

#         self.enc3 = DoubleConv(256, 512, dropout=True)
#         self.down3 = nn.Conv1d(512, 512, kernel_size=4, stride=2, padding=1)

#         self.enc4 = DoubleConv(512, 512, dropout=True)
#         self.down4 = nn.Conv1d(512, 512, kernel_size=4, stride=2, padding=1)

#         # 5階層目追加
#         self.enc5 = DoubleConv(512, 1024, dropout=True)
#         self.down5 = nn.Conv1d(1024, 1024, kernel_size=4, stride=2, padding=1)

#         # --- ボトルネック (BiLSTM) ---
#         # 入力サイズを1024に変更
#         self.bilstm = nn.LSTM(
#             input_size=1024,
#             hidden_size=1024,
#             num_layers=1,
#             bidirectional=True,
#             batch_first=True,
#         )
#         self.relu = nn.ReLU(inplace=False)

#         # --- デコーダ (5階層) ---
#         self.up1 = nn.ConvTranspose1d(
#             2048, 1024, kernel_size=4, stride=2, padding=1, output_padding=1
#         )
#         self.dec1 = DoubleConv(2048, 1024, dropout=True)  # 1024(up) + 1024(skip)

#         self.up2 = nn.ConvTranspose1d(1024, 512, kernel_size=4, stride=2, padding=1)
#         self.dec2 = DoubleConv(1024, 512, dropout=True)  # 512(up) + 512(skip)

#         self.up3 = nn.ConvTranspose1d(512, 256, kernel_size=4, stride=2, padding=1)
#         self.dec3 = DoubleConv(768, 256, dropout=True)  # 256(up) + 512(skip)

#         self.up4 = nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1)
#         self.dec4 = DoubleConv(384, 128, dropout=True)  # 128(up) + 256(skip)

#         # 5階層目追加
#         self.up5 = nn.ConvTranspose1d(128, 128, kernel_size=4, stride=2, padding=1)
#         self.dec5 = DoubleConv(256, 128, dropout=True)  # 128(up) + 128(skip)

#         # --- 出力層 ---
#         self.out_conv = nn.Conv1d(128, out_channels, kernel_size=1)

#     def forward(self, x):
#         # --- エンコーダ ---
#         s1 = self.enc1(x)
#         p1 = self.down1(s1)

#         s2 = self.enc2(p1)
#         p2 = self.down2(s2)

#         s3 = self.enc3(p2)
#         p3 = self.down3(s3)

#         s4 = self.enc4(p3)
#         p4 = self.down4(s4)

#         s5 = self.enc5(p4)
#         p5 = self.down5(s5)

#         # --- ボトルネック ---
#         b = p5.permute(0, 2, 1)
#         b, _ = self.bilstm(b)
#         b = self.relu(b)
#         b = b.permute(0, 2, 1)

#         # --- デコーダ ---
#         d1 = self.up1(b)
#         d1 = self.crop_and_concat(s5, d1)
#         d1 = self.dec1(d1)

#         d2 = self.up2(d1)
#         d2 = self.crop_and_concat(s4, d2)
#         d2 = self.dec2(d2)

#         d3 = self.up3(d2)
#         d3 = self.crop_and_concat(s3, d3)
#         d3 = self.dec3(d3)

#         d4 = self.up4(d3)
#         d4 = self.crop_and_concat(s2, d4)
#         d4 = self.dec4(d4)

#         d5 = self.up5(d4)
#         d5 = self.crop_and_concat(s1, d5)
#         d5 = self.dec5(d5)

#         return self.out_conv(d5)

#     def crop_and_concat(self, skip_tensor, up_tensor):
#         diff = skip_tensor.size(2) - up_tensor.size(2)
#         if diff > 0:
#             start = diff // 2
#             end = start + up_tensor.size(2)
#             skip_tensor = skip_tensor[:, :, start:end]

#         return torch.cat([skip_tensor, up_tensor], dim=1)
