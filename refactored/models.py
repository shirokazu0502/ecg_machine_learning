
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
        num_channels=15, # New argument
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
            num_channels, # Pass to Encoder
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
        num_channels, # New argument
    ):
        super().__init__()

        self.datalength = datalength
        self.conv_layer_sizes = conv_layer_sizes
        self.conditional = conditional
        self.num_channels = num_channels # Store num_channels

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
            x = torch.reshape(x, (-1, self.num_channels, self.datalength)) # Use self.num_channels

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


import torch.nn.functional as F

class UNet1D(nn.Module):
    def __init__(self, num_channels=15, out_channels=8, base_filters=32): # Changed in_channels to num_channels
        super(UNet1D, self).__init__()
        self.in_channels = num_channels # Use num_channels
        self.out_channels = out_channels

        self.e1 = DoubleConv1d(self.in_channels, base_filters) # Use self.in_channels
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.e2 = DoubleConv1d(base_filters, base_filters * 2)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.e3 = DoubleConv1d(base_filters * 2, base_filters * 4)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.e4 = DoubleConv1d(base_filters * 4, base_filters * 8)
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.bottleneck = DoubleConv1d(base_filters * 8, base_filters * 16)

        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="linear", align_corners=True),
            nn.Conv1d(base_filters * 16, base_filters * 8, kernel_size=3, padding=1),
        )
        self.d4 = DoubleConv1d(base_filters * 16, base_filters * 8)

        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="linear", align_corners=True),
            nn.Conv1d(base_filters * 8, base_filters * 4, kernel_size=3, padding=1),
        )
        self.d3 = DoubleConv1d(base_filters * 8, base_filters * 4)

        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="linear", align_corners=True),
            nn.Conv1d(base_filters * 4, base_filters * 2, kernel_size=3, padding=1),
        )
        self.d2 = DoubleConv1d(base_filters * 4, base_filters * 2)

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="linear", align_corners=True),
            nn.Conv1d(base_filters * 2, base_filters, kernel_size=3, padding=1),
        )
        self.d1 = DoubleConv1d(base_filters * 2, base_filters)

        self.out_conv = nn.Conv1d(base_filters, out_channels, kernel_size=1)

    def forward(self, x):
        s1 = self.e1(x)
        p1 = self.pool1(s1)
        s2 = self.e2(p1)
        p2 = self.pool2(s2)
        s3 = self.e3(p2)
        p3 = self.pool3(s3)
        s4 = self.e4(p3)
        p4 = self.pool4(s4)

        b = self.bottleneck(p4)

        u4 = self.up4(b)
        if u4.shape[2] != s4.shape[2]:
            u4 = F.interpolate(u4, size=s4.shape[2], mode="linear", align_corners=True)
        c4 = torch.cat([u4, s4], dim=1)
        d4 = self.d4(c4)

        u3 = self.up3(d4)
        if u3.shape[2] != s3.shape[2]:
            u3 = F.interpolate(u3, size=s3.shape[2], mode="linear", align_corners=True)
        c3 = torch.cat([u3, s3], dim=1)
        d3 = self.d3(c3)

        u2 = self.up2(d3)
        if u2.shape[2] != s2.shape[2]:
            u2 = F.interpolate(u2, size=s2.shape[2], mode="linear", align_corners=True)
        c2 = torch.cat([u2, s2], dim=1)
        d2 = self.d2(c2)

        u1 = self.up1(d2)
        if u1.shape[2] != s1.shape[2]:
            u1 = F.interpolate(u1, size=s1.shape[2], mode="linear", align_corners=True)
        c1 = torch.cat([u1, s1], dim=1)
        d1 = self.d1(c1)

        output = self.out_conv(d1)

        return output
