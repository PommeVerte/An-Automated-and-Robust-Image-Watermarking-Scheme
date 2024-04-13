import torch
import torch.nn as nn


class ConvBNRelu(nn.Module):   
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ConvBNRelu, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding='same'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
            # nn.ReLU() cannot set inplace=True as we need to backward twice,
            # because the network shouldn't be modified before the second backward.
        )

    def forward(self, x):
        return self.layers(x)


class ConvSigm(nn.Module):   
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ConvSigm, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding='same'),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)
        

class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InceptionModule, self).__init__()

        self.path1 = ConvBNRelu(in_channels, 32, (1,1))
        self.path2 = nn.Sequential(
            ConvBNRelu(in_channels, 32, (1,1)),
            ConvBNRelu(32, 32, (3,3))
        )
        self.path3 = nn.Sequential(
            ConvBNRelu(in_channels, 32, (1,1)),
            ConvBNRelu(32, 32, (3,3)),
            ConvBNRelu(32, 32, (3,3)),
        )
        self.total = ConvBNRelu(96, out_channels, (1,1))

    def forward(self, inputs):
        conv1 = self.path1(inputs)
        conv2 = self.path2(inputs)
        conv3 = self.path3(inputs)
        concat = torch.cat((conv1, conv2, conv3), dim=1)
        conv4 = self.total(concat)
        outputs = torch.add(inputs, conv4)
        return outputs


class InceptionImmediate(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InceptionImmediate, self).__init__()

        self.path1 = ConvBNRelu(in_channels, 32, (1,1))
        self.path2 = nn.Sequential(
            ConvBNRelu(in_channels, 32, (1,1)),
            ConvBNRelu(32, 32, (3,3))
        )
        self.path3 = nn.Sequential(
            ConvBNRelu(in_channels, 32, (1,1)),
            ConvBNRelu(32, 32, (3,3)),
            ConvBNRelu(32, 32, (3,3)),
        )
        self.total = ConvBNRelu(96, out_channels, (1,1))

    def forward(self, inputs):
        conv1 = self.path1(inputs)
        conv2 = self.path2(inputs)
        conv3 = self.path3(inputs)
        concat = torch.cat((conv1, conv2, conv3), dim=1)
        conv4 = self.total(concat)
        outputs = torch.add(inputs, conv4)
        return concat, conv4, outputs # B1, B2, outputs