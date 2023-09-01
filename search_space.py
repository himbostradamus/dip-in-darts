from collections import OrderedDict
import torch
import nni.retiarii.nn.pytorch as nn
from nni import trace
from nni.retiarii import model_wrapper
from nni.retiarii.nn.pytorch import Cell

from darts.common_utils import *

@trace
def conv_2d(C_in, C_out, kernel_size=3, dilation=1, padding=1, activation=None):
    return nn.Sequential(
        nn.Conv2d(C_in, C_out, kernel_size=kernel_size, dilation=dilation, padding=padding, bias=False),
        nn.BatchNorm2d(C_out),
        nn.ReLU() if activation is None else activation,
        nn.Conv2d(C_out, C_out, kernel_size=kernel_size, dilation=dilation, padding=padding, bias=False),
        nn.BatchNorm2d(C_out),
        nn.ReLU() if activation is None else activation
    )

@trace
def depthwise_separable_conv(C_in, C_out, kernel_size=3, dilation=1, padding=1, activation=None):
    return nn.Sequential(
        nn.Conv2d(C_in, C_in, kernel_size=kernel_size, dilation=dilation, padding=padding, groups=C_in, bias=False),
        nn.Conv2d(C_in, C_out, 1, bias=False),
        nn.BatchNorm2d(C_out),
        nn.ReLU() if activation is None else activation,
        nn.Conv2d(C_out, C_out, kernel_size=kernel_size, dilation=dilation, padding=padding, groups=C_out, bias=False),
        nn.Conv2d(C_out, C_out, 1, bias=False),
        nn.BatchNorm2d(C_out),
        nn.ReLU() if activation is None else activation
    )

def pools():
    pool_dict = OrderedDict([
        ("MaxPool2d", nn.MaxPool2d(kernel_size=2, stride=2, padding=0)),
        # ("AvgPool2d", nn.AvgPool2d(kernel_size=2, stride=2, padding=0)),
        # ("AdaMaxPool2d", nn.AdaptiveMaxPool2d(1)),
        # ("AdaAvgPool2d", nn.AdaptiveAvgPool2d(1)),
        # ("DepthToSpace", nn.PixelShuffle(2)),
    ])
    return pool_dict

def upsamples():
    upsample_dict = OrderedDict([
        ("Upsample_nearest", nn.Upsample(scale_factor=2, mode='nearest')),
        # ("Upsample_bilinear", nn.Upsample(scale_factor=2, mode='bilinear')),

    ])
    return upsample_dict

def convs(C_in, C_out):
    # all padding should follow this formula:
    # pd = (ks - 1) * dl // 2
    conv_dict = OrderedDict([
        
        # ("conv2d_1x1_Relu", conv_2d(C_in, C_out)),
        # ("conv2d_1x1_SiLU", conv_2d(C_in, C_out, activation=nn.SiLU())),

        # ("conv2d_3x3_Relu", conv_2d(C_in, C_out, kernel_size=3, padding=1)),
        ("conv2d_3x3_SiLU", conv_2d(C_in, C_out, kernel_size=3, padding=1, activation=nn.SiLU())),
        # ("conv2d_3x3_Relu_1dil", conv_2d(C_in, C_out, kernel_size=3, padding=2, dilation=2)),

        # ("conv2d_5x5_Relu", conv_2d(C_in, C_out, kernel_size=5, padding=2)),
        # ("conv2d_5x5_Relu_1dil", conv_2d(C_in, C_out, kernel_size=5, padding=4, dilation=2, activation=nn.SiLU())),
        # ("conv2d_5x5_SiLU", conv_2d(C_in, C_out, kernel_size=5, padding=2, activation=nn.SiLU())),


        # ("convDS_1x1_Relu", depthwise_separable_conv(C_in, C_out)),
        # ("convDS_1x1_SiLU", depthwise_separable_conv(C_in, C_out, activation=nn.SiLU())),

        # ("convDS_3x3_Relu", depthwise_separable_conv(C_in, C_out, kernel_size=3, padding=1)),
        # ("convDS_3x3_SiLU", depthwise_separable_conv(C_in, C_out, kernel_size=3, padding=1, activation=nn.SiLU())),

        # ("convDS_5x5_Relu", depthwise_separable_conv(C_in, C_out, kernel_size=5, padding=2)),
        # ("convDS_5x5_SiLU", depthwise_separable_conv(C_in, C_out, kernel_size=5, padding=2, activation=nn.SiLU())),
    ])
    return conv_dict

@trace
class Preprocessor(nn.Module):
    def __init__(self, C_in, C_out):
        super().__init__()

        self.conv1 = nn.Conv2d(C_in, C_out, 1)

    def forward(self, x):
        return [self.conv1(x[0])]

@trace
class Postprocessor(nn.Module):
    def __init__(self, C_in, C_out):
        super().__init__()

        self.conv1 = nn.Conv2d(C_in, C_out, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(C_in, C_out, kernel_size=3, padding=1)

    def forward(self, x):
        return [self.conv1(x[0]), self.conv2(x[1])]

@trace
@model_wrapper
class DARTS_UNet(nn.Module):
    def __init__(self, C_in=1, C_out=1, depth=4):
        super().__init__()

        # all padding should follow this formula:
        # pd = (ks - 1) * dl // 2
        self.pr = False
        self.depth = depth
        
        self.in_layer = nn.Conv2d(C_in, 64, kernel_size=3, padding=1)

        # Encoders
        filters = 64
        self.encoders = nn.ModuleList()
        for i in range(depth):
            self.encoders.append(Cell(pools(), num_nodes=1, num_ops_per_node=1, num_predecessors=1, label=f'pool_{i+1}'))
            self.encoders.append(Cell(convs(filters, filters*2), num_nodes=1, num_ops_per_node=1, num_predecessors=1, label=f'conv_{i+1}'))
            filters *= 2

        # Decoders
        self.decoders = nn.ModuleList()
        for i in range(depth):
            self.decoders.append(Cell(upsamples(), num_nodes=1, num_ops_per_node=1, num_predecessors=1, label=f'upsample_{i+1}'))
            filters //= 2
            self.decoders.append(Cell(convs(filters*3, filters), num_nodes=1, num_ops_per_node=1, num_predecessors=1, label=f'conv_{i+1+depth}'))

        self.out_layer = nn.Conv2d(64, C_out, kernel_size=3, padding=1)

    def forward(self, x):

        if self.pr:
            print(f'input shape: {x.shape}\n')

        x = self.in_layer(x)  # Apply the initial layer
        skip_connections = [x]

        for i in range(self.depth):
            x = self.encoders[2*i]([x])
            x = self.encoders[2*i+1]([x])
            skip_connections.append(x)

        for i in range(self.depth):
            upsampled = self.decoders[2*i]([x])
            cropped = self.crop_tensor(upsampled, skip_connections[-(i+2)])
            x = torch.cat([cropped, upsampled], 1)
            x = self.decoders[2*i+1]([x])

        x = self.out_layer(x)  # Apply the final layer

        return x

    def crop_tensor(self, target_tensor, tensor):
        target_size = target_tensor.size()[2]  # Assuming height and width are same
        tensor_size = tensor.size()[2]
        delta = tensor_size - target_size
        delta = delta // 2
        return tensor[:, :, delta:tensor_size-delta, delta:tensor_size-delta]
    
    def test(self):
        """
        This will input a random tensor of 1x1x128x128 and test the forward pass.
        """
        self.pr = True
        x = torch.randn(1, 1, 128, 128)
        y = self.forward(x)
        assert y.shape == (1, 1, 128, 128), "Output shape should be (1, 1, 128, 128), got {}".format(y.shape)
        print(f'output shape: {y.shape}\n')
        print("Test passed.\n\n")



def get_U_Net(in_channels=1, out_channels=1, init_features=64, pretrained=False):
    return torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
                       in_channels=in_channels, out_channels=out_channels, init_features=init_features, pretrained=pretrained)