from audioop import bias
import torch
import torch.nn as nn
from torchsummary import summary
import os
import pandas as pd
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHANNEL = 15


def idx2onehot(idx, n):
    idx = idx.to(device)
    assert torch.max(idx).item() < n
    if idx.dim() == 1:
        idx = idx.unsqueeze(1)

    onehot = torch.zeros(idx.size(0), n).to(device)
    onehot.scatter_(1, idx, 1)

    return onehot


class LSTM(nn.Module):

    def __init__(
        self,
        input_size,
        hidden1_size,
        hidden2_size,
        num_layers,
        output_size,
        dropout,
        conditional=False,
        look_back=20,
    ):
        super(LSTM, self).__init__()

        self.lstm_layers = nn.Sequential(
            nn.Conv1d(input_size, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.LSTM(128, hidden1_size, num_layers, batch_first=True, dropout=dropout),
            nn.LSTM(128, hidden2_size, num_layers, batch_first=True, dropout=dropout),
            nn.Dropout(dropout),
            nn.Linear(hidden2_size, output_size),
        )

        # 条件付きモデルの場合
        # self.conditional = conditional
        # if conditional:
        #     assert num_labels > 0
        #     self.label_fc = nn.Linear(num_labels, hidden1_size)

        # 全結合層（LSTMの出力を目的のサイズに変換）
        self.fc = nn.Linear(hidden1_size, output_size)

    def forward(self, x, c=None):
        batch_size = x.size(0)

        # 条件付きの場合、条件情報を入力に加える
        if self.conditional and c is not None:
            # 条件情報を結合
            c = self.label_fc(c)
            c = c.unsqueeze(1).expand(
                batch_size, x.size(1), -1
            )  # シーケンスの各ステップに同じ条件を加える
            x = torch.cat([x, c], dim=2)  # 特徴次元を結合

        # LSTMにデータを通す
        lstm_out, _ = self.lstm(x)

        # 最後の時間ステップの出力を使って全結合層に入力
        output = self.fc(lstm_out[:, -1, :])

        return output

    def inference(self, x, c=None):
        # 推論用のコード（再構築などは不要なためシンプル）
        return self.forward(x, c)


class Flatten(nn.Module):
    def forward(self, input):
        # print(input.size(1))
        return input.view(-1, input.size(1) * input.size(2))
        # return input.view(input.size(0), -1)
