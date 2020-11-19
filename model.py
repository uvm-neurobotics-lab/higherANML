import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, pooling=True):
        super(ConvBlock, self).__init__()
        self.pooling = pooling
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            stride=1,
            padding=0,
        )
        self.norm = nn.InstanceNorm2d(out_channels, affine=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = torch.relu(x)
        if self.pooling:
            x = F.max_pool2d(x, kernel_size=2, stride=2)
        return x


class RLN(nn.Module):
    def __init__(self, channels):
        super(RLN, self).__init__()
        self.convBlock1 = ConvBlock(1, channels)
        self.convBlock2 = ConvBlock(channels, channels)
        self.convBlock3 = ConvBlock(channels, channels, pooling=False)

    def forward(self, x):
        x = self.convBlock1(x)
        x = self.convBlock2(x)
        x = self.convBlock3(x)

        x = torch.flatten(x, start_dim=1)
        return x


class NM(nn.Module):
    def __init__(self, channels, mask_size):
        super(NM, self).__init__()
        self.convBlock1 = ConvBlock(1, channels)
        self.convBlock2 = ConvBlock(channels, channels)
        self.convBlock3 = ConvBlock(channels, channels, pooling=False)
        self.fc = nn.Linear(in_features=channels * 3 * 3, out_features=mask_size)

    def forward(self, x):
        x = self.convBlock1(x)
        x = self.convBlock2(x)
        x = self.convBlock3(x)

        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        x = torch.sigmoid(x)

        return x


class ANML(nn.Module):
    def __init__(self, rln_chs, nm_chs, mask_size):
        super(ANML, self).__init__()
        self.nm = NM(nm_chs, mask_size)
        self.rln = RLN(rln_chs)
        self.fc = nn.Linear(2304, 1000)

    def forward(self, x, modulation=True):

        features = self.rln(x)
        nm_mask = self.nm(x)

        features = features * nm_mask

        out = self.fc(features)

        return out
