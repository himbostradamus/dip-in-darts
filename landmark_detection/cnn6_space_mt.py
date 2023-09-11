
import numpy as np
from collections import OrderedDict
import torch
import nni.retiarii.nn.pytorch as nn
from nni import trace
from nni.retiarii import model_wrapper
from nni.retiarii.nn.pytorch import Cell, LayerChoice, InputChoice, ValueChoice

numOfLabels = 4
maxStage = 6

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

@trace
def attention_channel(channel, reduction=16):
    # this combined with the attention_forward method will perform channel attention
    return nn.Sequential(
        nn.Linear(channel, channel // reduction, bias=False),
        nn.ReLU(inplace=True),
        nn.Linear(channel // reduction, channel, bias=False),
        nn.Sigmoid()
    )

def pools():
    # these are your pooling layer choices
    pool_dict = OrderedDict([
        ("MaxPool2d", nn.MaxPool2d(kernel_size=2, stride=2, padding=0)),
        ("AvgPool2d", nn.AvgPool2d(kernel_size=2, stride=2, padding=0)),
        # ("MaxPool2d_3x3", nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        # ("AvgPool2d_3x3", nn.AvgPool2d(kernel_size=3, stride=2, padding=1)),
        # ("MaxPool2d_5x5", nn.MaxPool2d(kernel_size=5, stride=2, padding=2)),
        # ("AvgPool2d_5x5", nn.AvgPool2d(kernel_size=5, stride=2, padding=2)),
    ])
    return pool_dict

def convs(C_in, C_out):
    # these are your convolutional layer choices
    # uncomment to expand your search space

    # all padding should follow this formula:
    # pd = (ks - 1) * dl // 2
    conv_dict = OrderedDict([
        # ("Identity", nn.Identity()),

        ("conv2d_1x1_Relu", conv_2d(C_in, C_out, kernel_size=1, padding=0)),
        # ("conv2d_1x1_SiLU", conv_2d(C_in, C_out, kernel_size=1, padding=0, activation=nn.SiLU())),
        # ("conv2d_1x1_Sigmoid", conv_2d(C_in, C_out, kernel_size=1, padding=0, activation=nn.Sigmoid())),

        ("conv2d_3x3_Relu", conv_2d(C_in, C_out, kernel_size=3, padding=1)),
        # ("conv2d_3x3_SiLU", conv_2d(C_in, C_out, kernel_size=3, padding=1, activation=nn.SiLU())),
        # ("conv2d_3x3_Sigmoid", conv_2d(C_in, C_out, kernel_size=3, padding=1, activation=nn.Sigmoid())),
        # ("conv2d_3x3_Relu_1dil", conv_2d(C_in, C_out, kernel_size=3, padding=2, dilation=2)),

        ("conv2d_5x5_Relu", conv_2d(C_in, C_out, kernel_size=5, padding=2)),
        # ("conv2d_5x5_SiLU", conv_2d(C_in, C_out, kernel_size=5, padding=2, activation=nn.SiLU())),
        # ("conv2d_5x5_Sigmoid", conv_2d(C_in, C_out, kernel_size=5, padding=2, activation=nn.Sigmoid())),
        # ("conv2d_5x5_Relu_1dil", conv_2d(C_in, C_out, kernel_size=5, padding=4, dilation=2, activation=nn.SiLU())),

        ("conv2d_7x7_Relu", conv_2d(C_in, C_out, kernel_size=7, padding=3)),
        # ("conv2d_7x7_SiLU", conv_2d(C_in, C_out, kernel_size=7, padding=3, activation=nn.SiLU())),
        # ("conv2d_7x7_Sigmoid", conv_2d(C_in, C_out, kernel_size=7, padding=3, activation=nn.Sigmoid())),
        # ("conv2d_7x7_Relu_1dil", conv_2d(C_in, C_out, kernel_size=7, padding=6, dilation=2, activation=nn.SiLU())),

        ("convDS_1x1_Relu", depthwise_separable_conv(C_in, C_out)),
        # ("convDS_1x1_SiLU", depthwise_separable_conv(C_in, C_out, activation=nn.SiLU())),
        # ("convDS_1x1_Sigmoid", depthwise_separable_conv(C_in, C_out, activation=nn.Sigmoid())),

        ("convDS_3x3_Relu", depthwise_separable_conv(C_in, C_out, kernel_size=3, padding=1)),
        # ("convDS_3x3_SiLU", depthwise_separable_conv(C_in, C_out, kernel_size=3, padding=1, activation=nn.SiLU())),
        # ("convDS_3x3_Sigmoid", depthwise_separable_conv(C_in, C_out, kernel_size=3, padding=1, activation=nn.Sigmoid())),
        # ("convDS_3x3_Relu_1dil", depthwise_separable_conv(C_in, C_out, kernel_size=3, padding=2, dilation=2)),

        ("convDS_5x5_Relu", depthwise_separable_conv(C_in, C_out, kernel_size=5, padding=2)),
        # ("convDS_5x5_SiLU", depthwise_separable_conv(C_in, C_out, kernel_size=5, padding=2, activation=nn.SiLU())),
        # ("convDS_5x5_Sigmoid", depthwise_separable_conv(C_in, C_out, kernel_size=5, padding=2, activation=nn.Sigmoid())),
        # ("convDS_5x5_Relu_1dil", depthwise_separable_conv(C_in, C_out, kernel_size=5, padding=4, dilation=2, activation=nn.SiLU())),
    ])
    return conv_dict

@trace
@model_wrapper
class CNN_Space_MT(nn.Module):
    """
    input: 
        tensor of size (minibatchsize, numberOfChannels=1, Length=64, Width=64)
        
    output: 
        tensor of the same size
    """
    def __init__(
            self, 
            in_channels=1, 
            out_features=8, 
            depth=6,
            attention=False
            ):
        super().__init__()
    
        ### Manual Inputs ###
        self.attention = attention
        self.depth = depth

        ### NAS Inputs ###
        self.pool = LayerChoice(pools(), label="pooling_method")        
        self.progression_rate = ValueChoice([1, 2, 4], label="progression_rate")
        self.filter_size = ValueChoice([8, 16], label="filter_size")
        
        # create 2d convolutional layers
        self.convs = nn.ModuleList()
        self.channel_attention = nn.ModuleList() if self.attention else [None]

        # create the convolutional layers
        for i in range(self.depth):
            out_channels = self.filter_size * self.progression_rate ** i

            # self.convs.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding='same'))
            self.convs.append(LayerChoice(convs(in_channels, out_channels), label=f"convolution_{i+1}"))

            # add channel attention after the convolutional layer if specified
            if self.attention:
                self.channel_attention.append(attention_channel(out_channels))

            # update the number of input channels for the next layer
            in_channels = out_channels

        # create fully connected layers
        self.fullyConnected = nn.Sequential(
                                 nn.Linear(in_features=out_channels*4, out_features=out_channels*2, bias=True),
                                 nn.ReLU()
                                 )

        # create output layer
        self.output = nn.Linear(in_features=out_channels*2, out_features=out_features, bias=True)

    def forward(self, x):
        # apply convolutional layers
        for idx, conv in enumerate(self.convs):
            x = conv(x)
            if self.attention:
                x = self.attention_forward(x, self.channel_attention[idx])
            x = self.pool(x)

        # flatten and apply fully connected layers
        x = torch.flatten(x, start_dim=1)
        x = self.fullyConnected(x)

        # apply output layer
        x = self.output(x)
        return x
    
    def attention_forward(self, x, fcs):
        b, c, _, _ = x.size()
        y = nn.AdaptiveAvgPool2d(1)(x).view(b, c)
        y = fcs(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
    
    def test(self):
        # test that CNN-6 is working
        random_data = torch.rand((20, 1, 128, 128))
        result = self.forward(random_data)
        print(result.shape)

def WingLoss(label_pred, label_true):
    batch_size = label_pred.shape[0]
    label_size = label_pred.shape[1]
    loss = 0
    for b in range(batch_size):
        for l in range(label_size):
            loss += wing(label_pred[b, l] - label_true[b, l])
    return loss

def wing(x, w=10, eps=2):
    if abs(x) < w:
        return w * np.log(1 + abs(x) / eps)
    else:
        return abs(x) - w + w * np.log(1 + w / eps)