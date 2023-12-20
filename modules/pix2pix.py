import torch
import torch.nn as nn
import torch.nn.functional as F


class DiscCNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super(DiscCNNBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=4, stride=stride, padding=1, bias=False, padding_mode="reflect"),
            nn.BatchNorm2d(out_channels),
            #nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.conv(x)


class Discriminator(nn.Module):
    def __init__(self, in_channels, features=[64,128, 256, 512], out_channels=1):
        super(Discriminator, self).__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(
                in_channels,
                features[0], kernel_size=4, stride=2, padding=1, padding_mode="reflect"),
            nn.LeakyReLU(0.2)
        )
        self.layers = nn.ModuleList()
        in_channels = features[0]
        for feature in features[1:]:
            self.layers.append(
                DiscCNNBlock(in_channels, feature, stride=1 if feature == features[-1] else 2))
            in_channels = feature
        self.output = nn.Conv2d(
            in_channels, out_channels, kernel_size=4, stride=1, padding=1, padding_mode="reflect")

    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)
        x = self.initial(x)
        for layer in self.layers:
            x = layer(x)
        x = self.output(x)
        return x


class GenCNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, act="relu", use_dropout=False):
        super(GenCNNBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False, padding_mode="reflect")
            if down
            else nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            # For some reason the InstanceNorm2d not working. However most of the pix2pix SOTA work use Instance
            # Norm instead of batchNorm here. Probably better with ResNetGenBlock not Unet.
            #nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2) if act == 'leaky' else nn.ReLU()
        )
        self.use_dropout= use_dropout
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv(x)
        return self.dropout(x) if self.use_dropout else x


class Generator(nn.Module):
    def __init__(self, 
                 in_channels=3, 
                 features=64, 
                 n_layers=6, 
                 out_channels=None, 
                 n_aux_classes=1):
        super(Generator, self).__init__()
        out_channels = in_channels if out_channels is None else out_channels
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=4, stride=2, padding=1, padding_mode='reflect')
        )
        self.down_layers = nn.ModuleList()
        channel_multipliers = [min(8, 2**i) for i in range(n_layers)]
        in_multiplier = channel_multipliers[0]
        for out_multiplier in channel_multipliers[1:]:
            self.down_layers.append(GenCNNBlock(
                features*in_multiplier, features*out_multiplier, down=True, act='relu', use_dropout=False))
            in_multiplier = out_multiplier
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features*in_multiplier,
                      features*in_multiplier,
                      kernel_size=4,
                      stride=2,
                      padding=1,
                      padding_mode='reflect'),
            nn.ReLU(), # 1x1
            GenCNNBlock(
                features*in_multiplier, features*in_multiplier, down=False, act='relu', use_dropout=False),
        )
        self.up_layers = nn.ModuleList()
        for out_multiplier in channel_multipliers[::-1][1:]:
            self.up_layers.append(GenCNNBlock(
                features * in_multiplier * 2,
                features * out_multiplier,
                down=False, act='leaky', use_dropout=False))
            in_multiplier = out_multiplier
        assert n_aux_classes > 0 and isinstance(n_aux_classes, int), \
            'Number of auxillary classes must be greater than 0.'
        n_aux_classes = max(n_aux_classes, 1)
        self.output = nn.Sequential(
            nn.ConvTranspose2d(
                features*2, 
                out_channels * n_aux_classes, 
                kernel_size=4, 
                stride=2, 
                padding=1),
        )
        self.out_channels = out_channels
        self.n_aux_classes = n_aux_classes

    def forward(self, x):
        x = self.initial(x)
        encoder_outputs = [x]
        for layer in self.down_layers:
            x = layer(x)
            encoder_outputs.append(x)
        x = self.bottleneck(x)
        for layer in self.up_layers:
            x = layer(torch.cat((x, encoder_outputs.pop()), dim=1))
        # batch_size * (n_proposal * n_classes) * H * W
        x = self.output(torch.cat((x, encoder_outputs.pop()), dim=1))
        # batch_size * n_proposal * n_classes * H * W
        x = x.view(-1, self.out_channels, self.n_aux_classes, *x.size()[-2:])
        x_cls = x.mean(dim=-1).mean(dim=-1)
        return x.max(dim=2).values, x_cls


def interpolate_groups(g, ratio, mode, align_corners):
    batch_size, num_objects = g.shape[:2]
    g = F.interpolate(g.flatten(start_dim=0, end_dim=1),
                scale_factor=ratio, mode=mode, align_corners=align_corners)
    g = g.view(batch_size, num_objects, *g.shape[1:])
    return g


def upsample_groups(g, ratio=2, mode='bilinear', align_corners=False):
    return interpolate_groups(g, ratio, mode, align_corners)

def downsample_groups(g, ratio=1/2, mode='area', align_corners=None):
    return interpolate_groups(g, ratio, mode, align_corners)


class GConv2D(nn.Conv2d):
    def forward(self, g):
        batch_size, num_objects = g.shape[:2]
        g = super().forward(g.flatten(start_dim=0, end_dim=1))
        return g.view(batch_size, num_objects, *g.shape[1:])


class GlobalHiddenUpdater(nn.Module):
    # Used in the decoder, multi-scale feature + GRU
    def __init__(self, g_dim, kernel_size=7):
        super().__init__()
        self.hidden_dim = g_dim
        padding = (kernel_size - 1) // 2
        self.transform = nn.Conv2d(g_dim*2, g_dim*3,
                                   kernel_size=kernel_size, padding = padding, stride = 1)
        nn.init.xavier_normal_(self.transform.weight)

    def forward(self, g, h):
        g = torch.cat([g, h], 1)
        values = self.transform(g)
        forget_gate = torch.sigmoid(values[:,:self.hidden_dim])
        update_gate = torch.sigmoid(values[:,self.hidden_dim:self.hidden_dim*2])
        new_value = torch.tanh(values[:,self.hidden_dim*2:])
        new_h = forget_gate*h*(1-update_gate) + update_gate*new_value

        return new_h

if __name__ == '__main__':
    layer= GlobalHiddenUpdater(g_dim=4)
    g = torch.rand((3,4,5,5))
    h = torch.rand((3,4,5,5))
    r = layer(g, h)
    print(r)
