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
class CNN_Space_OS(nn.Module):
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
            progression_rate=2,
            filter_size=16,
            nodes=1,
            attention=False
            ):
        super().__init__()
    
        ### Manual Inputs ###
        self.attention = attention
        self.depth = depth
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)      
        self.progression_rate = progression_rate
        self.filter_size = filter_size
        self.nodes = nodes
        
        # create 2d convolutional layers
        self.convs = nn.ModuleList()
        self.postconvs = nn.ModuleList()
        self.channel_attention = nn.ModuleList() if self.attention else [None]

        self.first_layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=self.filter_size, kernel_size=3, padding='same'),
            nn.ReLU()
        )
        in_channels = self.filter_size

        # create the convolutional layers
        for i in range(self.depth):
            out_channels = self.filter_size * self.progression_rate ** i
           
            # here is the cell documentation: https://nni.readthedocs.io/en/stable/reference/nas/search_space.html#cell
            conv_candidates = convs(in_channels, in_channels)
            self.convs.append(Cell(
                    op_candidates=conv_candidates, 
                    num_nodes=self.nodes, 
                    num_ops_per_node=1,
                    num_predecessors=1,
                    label=f'conv_{i+1}',
                    ))

            # cells need the same dimensions for input and output, so to acheive the same dimensions, we add a postconvolutional layer
            # unfortunately this makes the model much larger and that may be a problem for lower resolution images
            # a model that is too big will overfit quickly
            self.postconvs.append(nn.Conv2d(in_channels=in_channels*self.nodes, out_channels=out_channels, kernel_size=3, padding='same'))

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
        print('entering forward')
        # apply first layer
        x = self.first_layer(x)

        # apply convolutional layers
        for i in range(self.depth):
            # inputs to cells must be a list
            print('entering cell')
            x = self.convs[i]([x])
            print('exiting cell')
            x = self.postconvs[i](x)
            if self.attention:
                x = self.attention_forward(x, self.channel_attention[i])
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