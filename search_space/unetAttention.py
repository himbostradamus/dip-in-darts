from collections import OrderedDict
import torch
import nni.retiarii.nn.pytorch as nn
from nni import trace
from nni.retiarii import model_wrapper
from nni.retiarii.nn.pytorch import Cell

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
        nn.Conv2d(C_out, C_out, kernel_size=kernel_size, dilation=dilation, padding=padding, groups=C_in, bias=False),
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
def attention(channel, reduction=16):
    return nn.Sequential(
        nn.Linear(channel, channel // reduction, bias=False),
        nn.ReLU(inplace=True),
        nn.Linear(channel // reduction, channel, bias=False),
        nn.Sigmoid()
    )


def pools():
    pool_dict = OrderedDict([
        ("MaxPool2d", nn.MaxPool2d(kernel_size=2, stride=2, padding=0)),
        ("AvgPool2d", nn.AvgPool2d(kernel_size=2, stride=2, padding=0)),
        ("AdaMaxPool2d", nn.AdaptiveMaxPool2d(1)),
        ("AdaAvgPool2d", nn.AdaptiveAvgPool2d(1)),
        ("MaxPool2d_3x3", nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ("AvgPool2d_3x3", nn.AvgPool2d(kernel_size=3, stride=2, padding=1)),
        ("MaxPool2d_5x5", nn.MaxPool2d(kernel_size=5, stride=2, padding=2)),
        ("AvgPool2d_5x5", nn.AvgPool2d(kernel_size=5, stride=2, padding=2)),
        ("MaxPool2d_7x7", nn.MaxPool2d(kernel_size=7, stride=2, padding=3)),
        ("AvgPool2d_7x7", nn.AvgPool2d(kernel_size=7, stride=2, padding=3)),
        ("MaxPool2d_9x9", nn.MaxPool2d(kernel_size=9, stride=2, padding=4)),
        ("AvgPool2d_9x9", nn.AvgPool2d(kernel_size=9, stride=2, padding=4)),

        # ("DepthToSpace", nn.PixelShuffle(2)),
    ])
    return pool_dict

def upsamples(C_in, C_out):
    upsample_dict = OrderedDict([
        ("Upsample_nearest", nn.Upsample(scale_factor=2, mode='nearest')),
        ("Upsample_bilinear", nn.Upsample(scale_factor=2, mode='bilinear')),
        
        ("TransConv_2x2_RelU", transposed_conv_2d(C_in, C_out, kernel_size=2, stride=2, padding=0)),
        ("TransConv_2x2_SiLU", transposed_conv_2d(C_in, C_out, kernel_size=2, stride=2, padding=0, activation=nn.SiLU())),
        ("TransConv_2x2_Sigmoid", transposed_conv_2d(C_in, C_out, kernel_size=2, stride=2, padding=0, activation=nn.Sigmoid())),

        ("TransConv_4x4_Relu", transposed_conv_2d(C_in, C_out)),
        ("TransConv_4x4_SiLU", transposed_conv_2d(C_in, C_out, activation=nn.SiLU())),
        ("TransConv_4x4_Sigmoid", transposed_conv_2d(C_in, C_out, activation=nn.Sigmoid())),
        
    ])
    return upsample_dict

def convs(C_in, C_out):
    # all padding should follow this formula:
    # pd = (ks - 1) * dl // 2
    conv_dict = OrderedDict([
        
        # ("Identity", nn.Identity()),

        # ("conv2d_1x1_Relu", conv_2d(C_in, C_out, kernel_size=1, padding=0)),
        # ("conv2d_1x1_SiLU", conv_2d(C_in, C_out, kernel_size=1, padding=0, activation=nn.SiLU())),
        ("conv2d_1x1_Sigmoid", conv_2d(C_in, C_out, kernel_size=1, padding=0, activation=nn.Sigmoid())),
        ("conv2d_1x1_Mish", conv_2d(C_in, C_out, kernel_size=1, padding=0, activation=nn.Mish())),

        ("conv2d_3x3_Relu", conv_2d(C_in, C_out, kernel_size=3, padding=1)),
        # ("conv2d_3x3_SiLU", conv_2d(C_in, C_out, kernel_size=3, padding=1, activation=nn.SiLU())),
        # ("conv2d_3x3_Sigmoid", conv_2d(C_in, C_out, kernel_size=3, padding=1, activation=nn.Sigmoid())),
        ("conv2d_3x3_Mish", conv_2d(C_in, C_out, kernel_size=3, padding=1, activation=nn.Mish())),
        # ("conv2d_3x3_Relu_1dil", conv_2d(C_in, C_out, kernel_size=3, padding=2, dilation=2)),
        # ("conv2d_3x3_SiLU_1dil", conv_2d(C_in, C_out, kernel_size=3, padding=2, dilation=2, activation=nn.SiLU())),
        # ("conv2d_3x3_Sigmoid_1dil", conv_2d(C_in, C_out, kernel_size=3, padding=2, dilation=2, activation=nn.Sigmoid())),

        # ("conv2d_5x5_Relu", conv_2d(C_in, C_out, kernel_size=5, padding=2)),
        ("conv2d_5x5_SiLU", conv_2d(C_in, C_out, kernel_size=5, padding=2, activation=nn.SiLU())),
        # ("conv2d_5x5_Sigmoid", conv_2d(C_in, C_out, kernel_size=5, padding=2, activation=nn.Sigmoid())),
        # ("conv2d_5x5_Mish", conv_2d(C_in, C_out, kernel_size=5, padding=2, activation=nn.Mish())),
        # ("conv2d_5x5_Relu_1dil", conv_2d(C_in, C_out, kernel_size=5, padding=4, dilation=2, activation=nn.SiLU())),
        # ("conv2d_5x5_SiLU_1dil", conv_2d(C_in, C_out, kernel_size=5, padding=4, dilation=2, activation=nn.SiLU())),
        # ("conv2d_5x5_Sigmoid_1dil", conv_2d(C_in, C_out, kernel_size=5, padding=4, dilation=2, activation=nn.Sigmoid())),
        # ("conv2d_5x5_Mish_1dil", conv_2d(C_in, C_out, kernel_size=5, padding=4, dilation=2, activation=nn.Mish())),

        # ("conv2d_7x7_Relu", conv_2d(C_in, C_out, kernel_size=7, padding=3)),
        # ("conv2d_7x7_SiLU", conv_2d(C_in, C_out, kernel_size=7, padding=3, activation=nn.SiLU())),
        # ("conv2d_7x7_Sigmoid", conv_2d(C_in, C_out, kernel_size=7, padding=3, activation=nn.Sigmoid())),
        ("conv2d_7x7_Mish", conv_2d(C_in, C_out, kernel_size=7, padding=3, activation=nn.Mish())),
        # ("conv2d_7x7_Relu_1dil", conv_2d(C_in, C_out, kernel_size=7, padding=6, dilation=2, activation=nn.SiLU())),
        # ("conv2d_7x7_SiLU_1dil", conv_2d(C_in, C_out, kernel_size=7, padding=6, dilation=2, activation=nn.SiLU())),
        # ("conv2d_7x7_Sigmoid_1dil", conv_2d(C_in, C_out, kernel_size=7, padding=6, dilation=2, activation=nn.Sigmoid())),

        # ("conv2d_9x9_Relu", conv_2d(C_in, C_out, kernel_size=9, padding=4)),
        # ("conv2d_9x9_SiLU", conv_2d(C_in, C_out, kernel_size=9, padding=4, activation=nn.SiLU())),
        # ("conv2d_9x9_Sigmoid", conv_2d(C_in, C_out, kernel_size=9, padding=4, activation=nn.Sigmoid())),
        # ("conv2d_9x9_Mish", conv_2d(C_in, C_out, kernel_size=9, padding=4, activation=nn.Mish())),
        # ("conv2d_9x9_Relu_1dil", conv_2d(C_in, C_out, kernel_size=9, padding=8, dilation=2, activation=nn.SiLU())),
        # ("conv2d_9x9_SiLU_1dil", conv_2d(C_in, C_out, kernel_size=9, padding=8, dilation=2, activation=nn.SiLU())),
        # ("conv2d_9x9_Sigmoid_1dil", conv_2d(C_in, C_out, kernel_size=9, padding=8, dilation=2, activation=nn.Sigmoid())),

        # ("convDS_1x1_Relu", depthwise_separable_conv(C_in, C_out)),
        ("convDS_1x1_SiLU", depthwise_separable_conv(C_in, C_out, activation=nn.SiLU())),
        # ("convDS_1x1_Sigmoid", depthwise_separable_conv(C_in, C_out, activation=nn.Sigmoid())),
        # ("convDS_1x1_Mish", depthwise_separable_conv(C_in, C_out, activation=nn.Mish())),

        # ("convDS_3x3_Relu", depthwise_separable_conv(C_in, C_out, kernel_size=3, padding=1)),
        # ("convDS_3x3_SiLU", depthwise_separable_conv(C_in, C_out, kernel_size=3, padding=1, activation=nn.SiLU())),
        # ("convDS_3x3_Sigmoid", depthwise_separable_conv(C_in, C_out, kernel_size=3, padding=1, activation=nn.Sigmoid())),
        # ("convDS_3x3_Relu_1dil", depthwise_separable_conv(C_in, C_out, kernel_size=3, padding=2, dilation=2)),
        # ("convDS_3x3_SiLU_1dil", depthwise_separable_conv(C_in, C_out, kernel_size=3, padding=2, dilation=2, activation=nn.SiLU())),
        # ("convDS_3x3_Sigmoid_1dil", depthwise_separable_conv(C_in, C_out, kernel_size=3, padding=2, dilation=2, activation=nn.Sigmoid())),

        # ("convDS_5x5_Relu", depthwise_separable_conv(C_in, C_out, kernel_size=5, padding=2)),
        # ("convDS_5x5_SiLU", depthwise_separable_conv(C_in, C_out, kernel_size=5, padding=2, activation=nn.SiLU())),
        # ("convDS_5x5_Sigmoid", depthwise_separable_conv(C_in, C_out, kernel_size=5, padding=2, activation=nn.Sigmoid())),
        ("convDS_5x5_Mish", depthwise_separable_conv(C_in, C_out, kernel_size=5, padding=2, activation=nn.Mish())),
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


import torch
import torch.nn as nn
import math

# Sinusoidal Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, max_len=64*64):
        super(PositionalEncoding, self).__init__()
        self.max_len = max_len

    def forward(self, x, d_model):
        pe = torch.zeros(self.max_len, d_model).to(x.device)
        position = torch.arange(0, self.max_len, dtype=torch.float).unsqueeze(1).to(x.device)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).to(x.device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        x = x + pe[:x.size(0), :x.size(2)]
        return x

class UNetWithAttention(nn.Module):
    def __init__(self, num_heads=2, features=128, in_channels=1, depth=4):
        super(UNetWithAttention, self).__init__()

        self.depth = depth

        # Assuming the embeddings size is the same as features for simplicity
        self.positional_encoding = PositionalEncoding()
        self.attention = nn.MultiheadAttention(embed_dim=features, num_heads=num_heads)

        # in layer
        mid_channels = 64
        self.in_layer = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)

        # encoders
        self.pools = nn.ModuleList()
        self.enc = nn.ModuleList()
        self.enc_attention = nn.ModuleList()
        for i in range(self.depth):
            self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
            self.enc.append(conv_2d(mid_channels, mid_channels*2, kernel_size=3, padding=1))
            self.enc_attention.append(nn.MultiheadAttention(embed_dim=mid_channels*2, num_heads=num_heads))
            mid_channels *= 2

        # decoders
        self.ups = nn.ModuleList()
        self.decs = nn.ModuleList()
        self.dec_attention = nn.ModuleList()
        for i in range(self.depth):
            self.ups.append(transposed_conv_2d(mid_channels, mid_channels, kernel_size=2, stride=2))
            mid_channels //= 2
            self.decs.append(conv_2d(mid_channels*3, mid_channels, kernel_size=3, padding=1))
            self.dec_attention.append(nn.MultiheadAttention(embed_dim=mid_channels, num_heads=num_heads))

        # out layer
        self.out_layer = nn.Conv2d(mid_channels, in_channels, kernel_size=1, padding=0)

        
    def forward(self, x):
        print(f'input shape: {x.shape}')
        # in layer
        x = self.in_layer(x)
        print(f'after in layer: {x.shape}')

        # skip connections
        skip_connections = [x]

        # encoders
        for i in range(self.depth):
            x = self.pools[i](x)
            print(f'after pool {i}: {x.shape}')
            x = self.enc[i](x)
            print(f'after enc {i}: {x.shape}\n')

            # apply attention to even encoders
            if i % 2 == 0:
                x = self.attention_forward(x, x.size(1), self.enc_attention[i])
                print(f'after attention {i}: {x.shape}\n\n')
            skip_connections.append(x)

        # decoders
        for i in range(self.depth):
            upsampled = self.ups[i](x)
            print(f'after upsample {i}: {upsampled.shape}')
            cropped = self.crop_tensor(upsampled, skip_connections[-(i+2)])
            x = torch.cat([cropped, upsampled], 1)
            x = self.decs[i](x)
            print(f'after dec {i}: {x.shape}\n')

            # apply attention to even decoders
            if i % 2 == 0:
                x = self.attention_forward(x, x.size(1), self.dec_attention[i])
                print(f'after attention {i}: {x.shape}\n\n')
        
        # out layer
        x = self.out_layer(x)

        return x
    
    def crop_tensor(self, target_tensor, tensor):
        target_size = target_tensor.size()[2]  # Assuming height and width are same
        tensor_size = tensor.size()[2]
        delta = tensor_size - target_size
        delta = delta // 2
        return tensor[:, :, delta:tensor_size-delta, delta:tensor_size-delta]

    def attention_forward(self, x, d_model, attention_layer):
        # Calculate the original height and width before flattening
        original_height, original_width = x.size(2), x.size(3)
        
        # Reshape for attention and add positional encodings
        x = x.view(x.size(0), x.size(1), -1).permute(2, 0, 1)  # [L, N, E]
        x = self.positional_encoding(x, d_model)
        
        # Apply attention
        attn_output, _ = attention_layer(x, x, x)
        
        # Reshape back to its original shape
        x = attn_output.permute(1, 2, 0).view(x.size(1), x.size(2), original_height, original_width)
        return x

    def test(self):
        x = torch.randn(1, 1, 64, 64)
        y = self.forward(x)
        print(y.shape)