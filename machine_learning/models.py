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
    ):

        super().__init__()

        if conditional:
            assert num_labels > 0

        assert type(enc_convlayer_sizes) == list
        assert type(enc_fclayer_sizes) == list
        assert type(dec_fclayer_sizes) == list
        assert type(dec_convlayer_sizes) == list
        assert type(latent_size) == int

        # self.layer_type = layer_type
        self.latent_size = latent_size
        self.datalength = datalength

        self.encoder = Encoder(
            datalength,
            enc_convlayer_sizes,
            enc_fclayer_sizes,
            latent_size,
            conditional,
            num_labels,
        ).to(device)
        self.decoder = Decoder(
            datalength,
            dec_convlayer_sizes,
            dec_fclayer_sizes,
            latent_size,
            conditional,
            num_labels,
        ).to(device)

        # self.encoder = Encoder(
        #    encoder_layer_sizes, latent_size, conditional, num_labels).to(device)
        # self.decoder = Decoder(
        #    decoder_layer_sizes, latent_size, conditional, num_labels).to(device)

    def forward(self, x, c=None):
        # print("x.shape")
        # print(x.shape)
        # print(x)
        # input()

        # if x.dim() > 2:
        #     #x = x.view(-1, 28*28)
        #     x = x.view(-1, self.datalength)

        batch_size = x.size(0)
        # print(batch_size)

        # print(x.shape)
        # summary(self.encoder, input_size=(1,15*384))
        means, log_var = self.encoder(x, c)
        # print("means")
        # print(means)
        # print("log_var")
        # print(log_var)
        # print("create_mean_log_var_z_csv")

        std = torch.exp(0.5 * log_var).to(device)
        eps = torch.randn([batch_size, self.latent_size]).to(device)
        # print(std.shape)
        # print(eps.shape)
        # print(means.shape)
        # print(std)
        # print(eps)
        # print(means)
        z = eps * std + means
        # print("z")
        # print(z)
        # df_means = pd.DataFrame(means.detach().to("cpu"))
        # df_var= pd.DataFrame(log_var.detach().to("cpu"))
        # df_z = pd.DataFrame(z.detach().to("cpu"))
        # df_con=pd.concat([df_means,df_var,df_z])
        #
        # df_con.to_csv('figs_newref/means_var_z.csv',index=False,header=False)

        recon_x = self.decoder(z, c)

        # print("recon_x.shape")
        # print(recon_x.shape)

        return recon_x, means, log_var, z

    # 使ってないかも
    def inference(self, n=1, c=None):

        batch_size = n
        z = torch.randn([batch_size, self.latent_size]).to(device)

        recon_x = self.decoder(z, c)

        return recon_x


class Flatten(nn.Module):
    def forward(self, input):
        # print(input.size(1))
        return input.view(-1, input.size(1) * input.size(2))
        # return input.view(input.size(0), -1)


# 本来はこっちのEncoder===================
# class Encoder(nn.Module):
#
#    def __init__(self, datalength, conv_layer_sizes, fc_layer_sizes, latent_size, conditional, num_labels):
#
#        super().__init__()
#
#        self.datalength = datalength
#        self.conv_layer_sizes = conv_layer_sizes
#
#        self.conditional = conditional
#        #if self.conditional:
#        #    layer_sizes[0] += num_labels
#
#        self.MLP = nn.Sequential().to(device)
#
#        if len(conv_layer_sizes) != 0:
#            for i, (conv_param_in, conv_param_out) in enumerate((zip(conv_layer_sizes[:-1], conv_layer_sizes[1:]))):# conv_layer_sizesの二個ずつペアで入力、出力で読み取ってる
#                self.MLP.add_module(name=f"AC{i}", module=nn.Conv1d(conv_param_in[0], conv_param_out[0], kernel_size=6, stride=conv_param_out[1], padding=2))#conv_param_in[0],conv_param_out[0]
#                self.MLP.add_module(name=f"AB{i}", module=nn.BatchNorm1d(conv_param_out[0]))
#                self.MLP.add_module(name=f"AA{i}", module=nn.ReLU())
#            self.MLP.add_module(name="F0", module=Flatten())
#        #self.MLP.add_module(name="F0", module=)
#        #self.MLP.add_module(name="AM0", module=nn.MaxPooling(2))
#        #self.MLP.add_module(name="AP0", module=nn.AdaptiveAvgPool1d(1))
#
#
#        for i, (in_size, out_size) in enumerate(zip(fc_layer_sizes[:-1], fc_layer_sizes[1:])):
#            self.MLP.add_module(
#                name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
#            self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())
#
#        self.linear_means = nn.Linear(fc_layer_sizes[-1], latent_size)
#        self.linear_log_var = nn.Linear(fc_layer_sizes[-1], latent_size)
#
#    def forward(self, x, c=None):
#
#        if len(self.conv_layer_sizes) != 0:
#            #x = torch.reshape(x, (-1, 2, self.datalength))
#            x = torch.reshape(x, (-1, CHANNEL, self.datalength))
#            #print(x.size())
#            #x = torch.reshape(x, (16, 1, 1800))
#
#        if self.conditional:
#            c = idx2onehot(c, n=10)
#            x = torch.cat((x, c), dim=-1)
#        # summary(self.decoder, input_size=(4,3))
#        x = self.MLP(x)
#
#
#        means = self.linear_means(x).to(device)
#        log_vars = self.linear_log_var(x).to(device)
#
#        return means, log_vars

# ===================


class Encoder(nn.Module):

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
        self.conv_layer_sizes = conv_layer_sizes

        self.conditional = conditional
        # if self.conditional:
        #    layer_sizes[0] += num_labels

        self.MLP_1 = nn.Sequential().to(device)
        self.MLP_2 = nn.Sequential().to(device)

        if len(conv_layer_sizes) != 0:
            for i, (conv_param_in, conv_param_out) in enumerate(
                (zip(conv_layer_sizes[:-1], conv_layer_sizes[1:]))
            ):  # conv_layer_sizesの二個ずつペアで入力、出力で読み取ってる
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
                )  # conv_param_in[0],conv_param_out[0]
                self.MLP_1.add_module(
                    name=f"AB{i}", module=nn.BatchNorm1d(conv_param_out[0])
                )
                self.MLP_1.add_module(name=f"AA{i}", module=nn.ReLU())
            self.MLP_1.add_module(name="F0", module=Flatten())
        # self.MLP.add_module(name="F0", module=)
        # self.MLP.add_module(name="AM0", module=nn.MaxPooling(2))
        # self.MLP.add_module(name="AP0", module=nn.AdaptiveAvgPool1d(1))

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
            # x = torch.reshape(x, (-1, 2, self.datalength))
            x = torch.reshape(x, (-1, CHANNEL, self.datalength))
            # print(x.size())
            # x = torch.reshape(x, (16, 1, 1800))

        if self.conditional:
            c = idx2onehot(c, n=10)
            x = torch.cat((x, c), dim=-1)
        # summary(self.decoder, input_size=(4,3))
        # print(x)
        # print(x.shape)
        # input("")

        x = self.MLP_1(x)
        # print("convlayer_output=============")
        # print(x)
        ## x_r = x.to(device)
        ##print(x.shape)
        # xx = torch.reshape(x, (1,-1))
        # df_x = pd.DataFrame(xx.detach().to("cpu"))
        # df_x.T.to_csv('figs_newref/convlayer_check.csv',index=False,header=False)
        x = self.MLP_2(x)

        means = self.linear_means(x).to(device)
        # print("means_x")
        # print(means)
        # print(x)

        log_vars = self.linear_log_var(x).to(device)
        # print("log_var_x")
        # print(log_vars)
        # print(x)

        return means, log_vars


class Reshape(nn.Module):
    def __init__(self, re_channel, re_length):
        super().__init__()
        self.re_channel = re_channel
        self.re_length = re_length

    def forward(self, input):
        # print(self.re_channel)
        # print(self.re_length)
        # print(input.shape)
        # print(torch.reshape(input, (-1, self.re_channel, self.re_length)).shape)
        return torch.reshape(input, (-1, self.re_channel, self.re_length))
        # return input.view(input.size(0), -1)


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

        print(fc_layer_sizes[-1])
        print(fc_layer_sizes)
        print("len(fc_layer_sizes)")
        print(len(fc_layer_sizes))

        self.conditional = conditional
        if self.conditional:
            input_size = latent_size + num_labels
        else:
            input_size = latent_size

            print("input_size")
            print(input_size)
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
                # self.MLP.add_module(name="ReLU", module=nn.ReLU())

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
                # self.MLP.add_module(name=f"AC{i}", module=nn.ConvTranspose1d(conv_param_in[0], conv_param_out[0], kernel_size=int(conv_param_in[1]), stride=int(conv_param_in[1]), padding=0))
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
                )  # ゼロからのCNNのところ見ながら調節
                # self.MLP.add_module(name=f"AB{i}", module=nn.BatchNorm1d(conv_param_out[0]))
                # self.MLP.add_module(name=f"AA{i}", module=nn.Sigmoid())
                # self.MLP.add_module(name="ReLU", module=nn.ReLU())
                # self.MLP.add_module(name="softplus", module=nn.Softplus())

    def forward(self, z, c):

        if self.conditional:
            c = idx2onehot(c, n=10).to(device)
            z = torch.cat((z, c), dim=-1)

        x = self.MLP(z)
        # print("decoder_x.shape")
        # print(x.shape)
        x = torch.reshape(x, (-1, 1, self.datalength))
        # print("decoder_x.reshape")
        # print(x.shape)
        # x = torch.reshape(x, (-1, 2, self.datalength))
        # x = torch.reshape(x, (16, 2, self.datalength))
        # print(x.shape)

        return x
