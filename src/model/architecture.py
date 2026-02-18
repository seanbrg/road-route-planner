import torch
import torch.nn as nn
import torchvision.transforms.functional as TF





"""
I       VII
II      VI
III     V
IV -> DBlock 

"""



class DoubleConv(nn.Module):
    """
    This function defines a double convolution layer with batch normalization and ReLU activation
    (Convolution => [BN] => ReLU) * 2
    Each Block in the encoder and decoder is a double convolution layer.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class DBlock(nn.Module):
    """
    The D-LinkNet Center Block.
    It uses cascaded dilated convolutions to expand the receptive field.
    This helps the model 'see' connections over longer distances (e.g. through shadows).
    """

    def __init__(self, channel):
        super().__init__()
        # Dilated convolutions with increasing rates
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=2, padding=2)
        self.dilate4 = nn.Conv2d(channel, channel, kernel_size=3, dilation=4, padding=4)
        self.dilate8 = nn.Conv2d(channel, channel, kernel_size=3, dilation=8, padding=8)

        # Batch Norms for stability
        self.bn = nn.BatchNorm2d(channel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Cascade logic: 1 -> 2 -> 4 -> 8
        d1 = self.relu(self.dilate1(x))
        d2 = self.relu(self.dilate2(d1))
        d4 = self.relu(self.dilate4(d2))
        d8 = self.relu(self.dilate8(d4))

        # Summation (Residual connection style) preserves information from all scales
        out = x + d1 + d2 + d4 + d8
        return out


class DL_UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder (Down)
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # D-LinkNet Bottleneck
        # First, expand features (512 -> 1024) like standard U-Net
        self.bottleneck_conv = DoubleConv(features[-1], features[-1] * 2)
        # Then, apply the D-Block to the high-level features
        self.dblock = DBlock(features[-1] * 2)

        # Decoder (Up)
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(feature * 2, feature))

        # Final 1x1 Conv
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # Down path
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottleneck (Standard Conv -> D-Block)
        x = self.bottleneck_conv(x)
        x = self.dblock(x)

        # Flip skip connections
        skip_connections = skip_connections[::-1]

        # Up path
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        return self.final_conv(x)