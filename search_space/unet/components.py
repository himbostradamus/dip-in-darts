from collections import OrderedDict
import nni.retiarii.nn.pytorch as nn
from nni import trace
import torch
import torch.nn.functional as F

@trace
def conv_2d(C_in, C_out, kernel_size=3, dilation=1, padding=1, activation=None):
    return nn.Sequential(
        nn.Conv2d(C_in, C_out, kernel_size=kernel_size, dilation=dilation, padding=padding, bias=False),
        nn.BatchNorm2d(C_out),
        nn.ReLU() if activation is None else activation,
        nn.Conv2d(C_out, C_out, kernel_size=kernel_size, dilation=dilation, padding=padding, bias=False),
        nn.BatchNorm2d(C_out),
        nn.ReLU() if activation is None else activation,
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
        nn.ReLU() if activation is None else activation,
    )

@trace
def transposed_conv_2d(C_in, C_out, kernel_size=4, stride=2, padding=1, activation=None):
    return nn.Sequential(
        nn.ConvTranspose2d(C_in, C_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(C_out),
        nn.ReLU() if activation is None else activation
    )

@trace
def seBlock(channel, reduction=16):
    return nn.Sequential(
        nn.Linear(channel, channel // reduction, bias=False),
        nn.ReLU(inplace=True),
        nn.Linear(channel // reduction, channel, bias=False),
        nn.Sigmoid()
    )

    # corresponding forward function 
   
def seForward(x, fcs):
    b, c, _, _ = x.size()
    y = nn.AdaptiveAvgPool2d(1)(x).view(b, c)
    y = fcs(y).view(b, c, 1, 1)
    return x * y.expand_as(x)


def pools():
    pool_dict = OrderedDict([
        ("MaxPool2d", nn.MaxPool2d(kernel_size=2, stride=2, padding=0)),
        ("AvgPool2d", nn.AvgPool2d(kernel_size=2, stride=2, padding=0)),
        # ("MaxPool2d_3x3", nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ("AvgPool2d_3x3", nn.AvgPool2d(kernel_size=3, stride=2, padding=1)),
        ("MaxPool2d_5x5", nn.MaxPool2d(kernel_size=5, stride=2, padding=2)),
        # ("AvgPool2d_5x5", nn.AvgPool2d(kernel_size=5, stride=2, padding=2)),
        # ("MaxPool2d_7x7", nn.MaxPool2d(kernel_size=7, stride=2, padding=3)),
        # ("AvgPool2d_7x7", nn.AvgPool2d(kernel_size=7, stride=2, padding=3)),
        # ("MaxPool2d_9x9", nn.MaxPool2d(kernel_size=9, stride=2, padding=4)),
        # ("AvgPool2d_9x9", nn.AvgPool2d(kernel_size=9, stride=2, padding=4)),

        # ("DepthToSpace", nn.PixelShuffle(2)),
        # ("AdaMaxPool2d", nn.AdaptiveMaxPool2d(1)),
        # ("AdaAvgPool2d", nn.AdaptiveAvgPool2d(1)),
    ])
    return pool_dict

def upsamples(C_in, C_out):
    upsample_dict = OrderedDict([
        ("Upsample_nearest", nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(C_in, C_out, kernel_size=1)
        )),
        ("Upsample_bilinear", nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(C_in, C_out, kernel_size=1)
        )),
        
        ("TransConv_2x2_RelU", transposed_conv_2d(C_in, C_out, kernel_size=2, stride=2, padding=0)),
        ("TransConv_2x2_SiLU", transposed_conv_2d(C_in, C_out, kernel_size=2, stride=2, padding=0, activation=nn.SiLU())),
        # ("TransConv_2x2_Sigmoid", transposed_conv_2d(C_in, C_out, kernel_size=2, stride=2, padding=0, activation=nn.Sigmoid())),

        # ("TransConv_4x4_Relu", transposed_conv_2d(C_in, C_out)),
        # ("TransConv_4x4_SiLU", transposed_conv_2d(C_in, C_out, activation=nn.SiLU())),
        # ("TransConv_4x4_Sigmoid", transposed_conv_2d(C_in, C_out, activation=nn.Sigmoid())),
        
    ])
    return upsample_dict

def convs(C_in, C_out):
    # all padding should follow this formula:
    # padding = (kernel_size - 1) * dilation // 2
    conv_dict = OrderedDict([
        
        # ("Identity", nn.Identity()),

        ("conv2d_1x1_Relu", conv_2d(C_in, C_out, kernel_size=1, padding=0)),
        ("conv2d_1x1_SiLU", conv_2d(C_in, C_out, kernel_size=1, padding=0, activation=nn.SiLU())),
        # ("conv2d_1x1_Sigmoid", conv_2d(C_in, C_out, kernel_size=1, padding=0, activation=nn.Sigmoid())),
        ("conv2d_1x1_Mish", conv_2d(C_in, C_out, kernel_size=1, padding=0, activation=nn.Mish())),

        ("conv2d_3x3_Relu", conv_2d(C_in, C_out, kernel_size=3, padding=1)),
        ("conv2d_3x3_SiLU", conv_2d(C_in, C_out, kernel_size=3, padding=1, activation=nn.SiLU())),
        # ("conv2d_3x3_Sigmoid", conv_2d(C_in, C_out, kernel_size=3, padding=1, activation=nn.Sigmoid())),
        ("conv2d_3x3_Mish", conv_2d(C_in, C_out, kernel_size=3, padding=1, activation=nn.Mish())),
        # ("conv2d_3x3_Relu_1dil", conv_2d(C_in, C_out, kernel_size=3, padding=2, dilation=2)),
        ("conv2d_3x3_SiLU_1dil", conv_2d(C_in, C_out, kernel_size=3, padding=2, dilation=2, activation=nn.SiLU())),
        # ("conv2d_3x3_Sigmoid_1dil", conv_2d(C_in, C_out, kernel_size=3, padding=2, dilation=2, activation=nn.Sigmoid())),

        # ("conv2d_5x5_Relu", conv_2d(C_in, C_out, kernel_size=5, padding=2)),
        # ("conv2d_5x5_SiLU", conv_2d(C_in, C_out, kernel_size=5, padding=2, activation=nn.SiLU())),
        # ("conv2d_5x5_Sigmoid", conv_2d(C_in, C_out, kernel_size=5, padding=2, activation=nn.Sigmoid())),
        # ("conv2d_5x5_Mish", conv_2d(C_in, C_out, kernel_size=5, padding=2, activation=nn.Mish())),
        # ("conv2d_5x5_Relu_1dil", conv_2d(C_in, C_out, kernel_size=5, padding=4, dilation=2, activation=nn.SiLU())),
        # ("conv2d_5x5_SiLU_1dil", conv_2d(C_in, C_out, kernel_size=5, padding=4, dilation=2, activation=nn.SiLU())),
        # ("conv2d_5x5_Sigmoid_1dil", conv_2d(C_in, C_out, kernel_size=5, padding=4, dilation=2, activation=nn.Sigmoid())),
        # ("conv2d_5x5_Mish_1dil", conv_2d(C_in, C_out, kernel_size=5, padding=4, dilation=2, activation=nn.Mish())),

        # ("conv2d_7x7_Relu", conv_2d(C_in, C_out, kernel_size=7, padding=3)),
        ("conv2d_7x7_SiLU", conv_2d(C_in, C_out, kernel_size=7, padding=3, activation=nn.SiLU())),
        # ("conv2d_7x7_Sigmoid", conv_2d(C_in, C_out, kernel_size=7, padding=3, activation=nn.Sigmoid())),
        # ("conv2d_7x7_Mish", conv_2d(C_in, C_out, kernel_size=7, padding=3, activation=nn.Mish())),
        # ("conv2d_7x7_Relu_1dil", conv_2d(C_in, C_out, kernel_size=7, padding=6, dilation=2, activation=nn.SiLU())),
        # ("conv2d_7x7_SiLU_1dil", conv_2d(C_in, C_out, kernel_size=7, padding=6, dilation=2, activation=nn.SiLU())),
        # ("conv2d_7x7_Sigmoid_1dil", conv_2d(C_in, C_out, kernel_size=7, padding=6, dilation=2, activation=nn.Sigmoid())),

        ("conv2d_9x9_Relu", conv_2d(C_in, C_out, kernel_size=9, padding=4)),
        # ("conv2d_9x9_SiLU", conv_2d(C_in, C_out, kernel_size=9, padding=4, activation=nn.SiLU())),
        # ("conv2d_9x9_Sigmoid", conv_2d(C_in, C_out, kernel_size=9, padding=4, activation=nn.Sigmoid())),
        # ("conv2d_9x9_Mish", conv_2d(C_in, C_out, kernel_size=9, padding=4, activation=nn.Mish())),
        # ("conv2d_9x9_Relu_1dil", conv_2d(C_in, C_out, kernel_size=9, padding=8, dilation=2, activation=nn.SiLU())),
        # ("conv2d_9x9_SiLU_1dil", conv_2d(C_in, C_out, kernel_size=9, padding=8, dilation=2, activation=nn.SiLU())),
        # ("conv2d_9x9_Sigmoid_1dil", conv_2d(C_in, C_out, kernel_size=9, padding=8, dilation=2, activation=nn.Sigmoid())),

        # ("convDS_1x1_Relu", depthwise_separable_conv(C_in, C_out)),
        # ("convDS_1x1_SiLU", depthwise_separable_conv(C_in, C_out, activation=nn.SiLU())),
        # ("convDS_1x1_Sigmoid", depthwise_separable_conv(C_in, C_out, activation=nn.Sigmoid())),
        # ("convDS_1x1_Mish", depthwise_separable_conv(C_in, C_out, activation=nn.Mish())),

        # ("convDS_3x3_Relu", depthwise_separable_conv(C_in, C_out, kernel_size=3, padding=1)),
        # ("convDS_3x3_SiLU", depthwise_separable_conv(C_in, C_out, kernel_size=3, padding=1, activation=nn.SiLU())),
        # ("convDS_3x3_Sigmoid", depthwise_separable_conv(C_in, C_out, kernel_size=3, padding=1, activation=nn.Sigmoid())),
        # ("convDS_3x3_Relu_1dil", depthwise_separable_conv(C_in, C_out, kernel_size=3, padding=2, dilation=2)),
        ("convDS_3x3_SiLU_1dil", depthwise_separable_conv(C_in, C_out, kernel_size=3, padding=2, dilation=2, activation=nn.SiLU())),
        # ("convDS_3x3_Sigmoid_1dil", depthwise_separable_conv(C_in, C_out, kernel_size=3, padding=2, dilation=2, activation=nn.Sigmoid())),

        ("convDS_5x5_Relu", depthwise_separable_conv(C_in, C_out, kernel_size=5, padding=2)),
        # ("convDS_5x5_SiLU", depthwise_separable_conv(C_in, C_out, kernel_size=5, padding=2, activation=nn.SiLU())),
        # ("convDS_5x5_Sigmoid", depthwise_separable_conv(C_in, C_out, kernel_size=5, padding=2, activation=nn.Sigmoid())),
        # ("convDS_5x5_Mish", depthwise_separable_conv(C_in, C_out, kernel_size=5, padding=2, activation=nn.Mish())),
        # ("convDS_5x5_Relu_1dil", depthwise_separable_conv(C_in, C_out, kernel_size=5, padding=4, dilation=2, activation=nn.SiLU())),
        # ("convDS_5x5_SiLU_1dil", depthwise_separable_conv(C_in, C_out, kernel_size=5, padding=4, dilation=2, activation=nn.SiLU())),
        # ("convDS_5x5_Sigmoid_1dil", depthwise_separable_conv(C_in, C_out, kernel_size=5, padding=4, dilation=2, activation=nn.Sigmoid())),

        # ("convDS_7x7_Relu", depthwise_separable_conv(C_in, C_out, kernel_size=7, padding=3)),
        # ("convDS_7x7_SiLU", depthwise_separable_conv(C_in, C_out, kernel_size=7, padding=3, activation=nn.SiLU())),
        # ("convDS_7x7_Sigmoid", depthwise_separable_conv(C_in, C_out, kernel_size=7, padding=3, activation=nn.Sigmoid())),
        # ("convDS_7x7_Relu_1dil", depthwise_separable_conv(C_in, C_out, kernel_size=7, padding=6, dilation=2, activation=nn.SiLU())),
        # ("convDS_7x7_SiLU_1dil", depthwise_separable_conv(C_in, C_out, kernel_size=7, padding=6, dilation=2, activation=nn.SiLU())),
        # ("convDS_7x7_Sigmoid_1dil", depthwise_separable_conv(C_in, C_out, kernel_size=7, padding=6, dilation=2, activation=nn.Sigmoid())),

        # ("convDS_9x9_Relu", depthwise_separable_conv(C_in, C_out, kernel_size=9, padding=4)),
        # ("convDS_9x9_SiLU", depthwise_separable_conv(C_in, C_out, kernel_size=9, padding=4, activation=nn.SiLU())),
        # ("convDS_9x9_Sigmoid", depthwise_separable_conv(C_in, C_out, kernel_size=9, padding=4, activation=nn.Sigmoid())),
        # ("convDS_9x9_Relu_1dil", depthwise_separable_conv(C_in, C_out, kernel_size=9, padding=8, dilation=2, activation=nn.SiLU())),
        # ("convDS_9x9_SiLU_1dil", depthwise_separable_conv(C_in, C_out, kernel_size=9, padding=8, dilation=2, activation=nn.SiLU())),
        # ("convDS_9x9_Sigmoid_1dil", depthwise_separable_conv(C_in, C_out, kernel_size=9, padding=8, dilation=2, activation=nn.Sigmoid())),
    ])
    return conv_dict

        ### 
        ### 
        ### this works well
        ### 
        ### 

        # ("conv2d_1x1_Relu", conv_2d(C_in, C_out, kernel_size=1, padding=0)),
        # ("conv2d_1x1_SiLU", conv_2d(C_in, C_out, kernel_size=1, padding=0, activation=nn.SiLU())),
        # ("conv2d_1x1_Mish", conv_2d(C_in, C_out, kernel_size=1, padding=0, activation=nn.Mish())),

        # ("conv2d_3x3_Relu", conv_2d(C_in, C_out, kernel_size=3, padding=1)),
        # ("conv2d_3x3_SiLU", conv_2d(C_in, C_out, kernel_size=3, padding=1, activation=nn.SiLU())),
        # ("conv2d_3x3_Mish", conv_2d(C_in, C_out, kernel_size=3, padding=1, activation=nn.Mish())),
        
        # ("conv2d_3x3_SiLU_1dil", conv_2d(C_in, C_out, kernel_size=3, padding=2, dilation=2, activation=nn.SiLU())),
        # ("convDS_3x3_SiLU_1dil", depthwise_separable_conv(C_in, C_out, kernel_size=3, padding=2, dilation=2, activation=nn.SiLU())),
        
        # ("convDS_5x5_Relu", depthwise_separable_conv(C_in, C_out, kernel_size=5, padding=2)),
        # ("conv2d_7x7_SiLU", conv_2d(C_in, C_out, kernel_size=7, padding=3, activation=nn.SiLU())),
        # ("conv2d_9x9_Relu", conv_2d(C_in, C_out, kernel_size=9, padding=4)),