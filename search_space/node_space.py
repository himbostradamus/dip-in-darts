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
        # ("AvgPool2d_3x3", nn.AvgPool2d(kernel_size=3, stride=2, padding=1)),
        # ("MaxPool2d_5x5", nn.MaxPool2d(kernel_size=5, stride=2, padding=2)),
        ("AvgPool2d_5x5", nn.AvgPool2d(kernel_size=5, stride=2, padding=2)),
        # ("MaxPool2d_7x7", nn.MaxPool2d(kernel_size=7, stride=2, padding=3)),
        # ("AvgPool2d_7x7", nn.AvgPool2d(kernel_size=7, stride=2, padding=3)),
        # ("MaxPool2d_9x9", nn.MaxPool2d(kernel_size=9, stride=2, padding=4)),
        # ("AvgPool2d_9x9", nn.AvgPool2d(kernel_size=9, stride=2, padding=4)),

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
        ("conv2d_3x3_SiLU_1dil", conv_2d(C_in, C_out, kernel_size=3, padding=2, dilation=2, activation=nn.SiLU())),
        # ("conv2d_3x3_Sigmoid_1dil", conv_2d(C_in, C_out, kernel_size=3, padding=2, dilation=2, activation=nn.Sigmoid())),
        # ("conv2d_3x3_Mish_1dil", conv_2d(C_in, C_out, kernel_size=3, padding=2, dilation=2, activation=nn.Mish())),

        # ("conv2d_5x5_Relu", conv_2d(C_in, C_out, kernel_size=5, padding=2)),
        # ("conv2d_5x5_SiLU", conv_2d(C_in, C_out, kernel_size=5, padding=2, activation=nn.SiLU())),
        # ("conv2d_5x5_Sigmoid", conv_2d(C_in, C_out, kernel_size=5, padding=2, activation=nn.Sigmoid())),
        ("conv2d_5x5_Mish", conv_2d(C_in, C_out, kernel_size=5, padding=2, activation=nn.Mish())),
        # ("conv2d_5x5_Relu_1dil", conv_2d(C_in, C_out, kernel_size=5, padding=4, dilation=2, activation=nn.SiLU())),
        # ("conv2d_5x5_SiLU_1dil", conv_2d(C_in, C_out, kernel_size=5, padding=4, dilation=2, activation=nn.SiLU())),
        # ("conv2d_5x5_Sigmoid_1dil", conv_2d(C_in, C_out, kernel_size=5, padding=4, dilation=2, activation=nn.Sigmoid())),
        # ("conv2d_5x5_Mish_1dil", conv_2d(C_in, C_out, kernel_size=5, padding=4, dilation=2, activation=nn.Mish())),
        # ("conv2d_5x5_Relu_2dil", conv_2d(C_in, C_out, kernel_size=5, padding=8, dilation=4, activation=nn.SiLU())),
        # ("conv2d_5x5_SiLU_2dil", conv_2d(C_in, C_out, kernel_size=5, padding=8, dilation=4, activation=nn.SiLU())),
        # ("conv2d_5x5_Sigmoid_2dil", conv_2d(C_in, C_out, kernel_size=5, padding=8, dilation=4, activation=nn.Sigmoid())),
        # ("conv2d_5x5_Mish_2dil", conv_2d(C_in, C_out, kernel_size=5, padding=8, dilation=4, activation=nn.Mish())),

        # ("conv2d_7x7_Relu", conv_2d(C_in, C_out, kernel_size=7, padding=3)),
        # ("conv2d_7x7_SiLU", conv_2d(C_in, C_out, kernel_size=7, padding=3, activation=nn.SiLU())),
        # ("conv2d_7x7_Sigmoid", conv_2d(C_in, C_out, kernel_size=7, padding=3, activation=nn.Sigmoid())),
        # ("conv2d_7x7_Mish", conv_2d(C_in, C_out, kernel_size=7, padding=3, activation=nn.Mish())),
        # ("conv2d_7x7_Relu_1dil", conv_2d(C_in, C_out, kernel_size=7, padding=6, dilation=2, activation=nn.SiLU())),
        # ("conv2d_7x7_SiLU_1dil", conv_2d(C_in, C_out, kernel_size=7, padding=6, dilation=2, activation=nn.SiLU())),
        # ("conv2d_7x7_Sigmoid_1dil", conv_2d(C_in, C_out, kernel_size=7, padding=6, dilation=2, activation=nn.Sigmoid())),
        # ("conv2d_7x7_Mish_1dil", conv_2d(C_in, C_out, kernel_size=7, padding=6, dilation=2, activation=nn.Mish())),
        # ("conv2d_7x7_Relu_2dil", conv_2d(C_in, C_out, kernel_size=7, padding=12, dilation=4, activation=nn.SiLU())),
        # ("conv2d_7x7_SiLU_2dil", conv_2d(C_in, C_out, kernel_size=7, padding=12, dilation=4, activation=nn.SiLU())),
        # ("conv2d_7x7_Sigmoid_2dil", conv_2d(C_in, C_out, kernel_size=7, padding=12, dilation=4, activation=nn.Sigmoid())),
        # ("conv2d_7x7_Mish_2dil", conv_2d(C_in, C_out, kernel_size=7, padding=12, dilation=4, activation=nn.Mish())),


        ("conv2d_9x9_Relu", conv_2d(C_in, C_out, kernel_size=9, padding=4)),
        # ("conv2d_9x9_SiLU", conv_2d(C_in, C_out, kernel_size=9, padding=4, activation=nn.SiLU())),
        # ("conv2d_9x9_Sigmoid", conv_2d(C_in, C_out, kernel_size=9, padding=4, activation=nn.Sigmoid())),
        # ("conv2d_9x9_Mish", conv_2d(C_in, C_out, kernel_size=9, padding=4, activation=nn.Mish())),
        # ("conv2d_9x9_Relu_1dil", conv_2d(C_in, C_out, kernel_size=9, padding=8, dilation=2, activation=nn.SiLU())),
        # ("conv2d_9x9_SiLU_1dil", conv_2d(C_in, C_out, kernel_size=9, padding=8, dilation=2, activation=nn.SiLU())),
        # ("conv2d_9x9_Sigmoid_1dil", conv_2d(C_in, C_out, kernel_size=9, padding=8, dilation=2, activation=nn.Sigmoid())),
        # ("conv2d_9x9_Mish_1dil", conv_2d(C_in, C_out, kernel_size=9, padding=8, dilation=2, activation=nn.Mish())),

        # ("conv2d_11x11_Relu", conv_2d(C_in, C_out, kernel_size=11, padding=5)),
        # ("conv2d_11x11_SiLU", conv_2d(C_in, C_out, kernel_size=11, padding=5, activation=nn.SiLU())),
        # ("conv2d_11x11_Sigmoid", conv_2d(C_in, C_out, kernel_size=11, padding=5, activation=nn.Sigmoid())),
        # ("conv2d_11x11_Mish", conv_2d(C_in, C_out, kernel_size=11, padding=5, activation=nn.Mish())),
        # ("conv2d_11x11_Relu_1dil", conv_2d(C_in, C_out, kernel_size=11, padding=10, dilation=2, activation=nn.SiLU())),
        # ("conv2d_11x11_SiLU_1dil", conv_2d(C_in, C_out, kernel_size=11, padding=10, dilation=2, activation=nn.SiLU())),
        # ("conv2d_11x11_Sigmoid_1dil", conv_2d(C_in, C_out, kernel_size=11, padding=10, dilation=2, activation=nn.Sigmoid())),
        # ("conv2d_11x11_Mish_1dil", conv_2d(C_in, C_out, kernel_size=11, padding=10, dilation=2, activation=nn.Mish())),

        # ("convDS_1x1_Relu", depthwise_separable_conv(C_in, C_out)),
        # ("convDS_1x1_SiLU", depthwise_separable_conv(C_in, C_out, activation=nn.SiLU())),
        # ("convDS_1x1_Sigmoid", depthwise_separable_conv(C_in, C_out, activation=nn.Sigmoid())),
        # ("convDS_1x1_Mish", depthwise_separable_conv(C_in, C_out, activation=nn.Mish())),

        ("convDS_3x3_Relu", depthwise_separable_conv(C_in, C_out, kernel_size=3, padding=1)),
        ("convDS_3x3_SiLU", depthwise_separable_conv(C_in, C_out, kernel_size=3, padding=1, activation=nn.SiLU())),
        # ("convDS_3x3_Sigmoid", depthwise_separable_conv(C_in, C_out, kernel_size=3, padding=1, activation=nn.Sigmoid())),
        # ("convDS_3x3_Mish", depthwise_separable_conv(C_in, C_out, kernel_size=3, padding=1, activation=nn.Mish())),
        # ("convDS_3x3_Relu_1dil", depthwise_separable_conv(C_in, C_out, kernel_size=3, padding=2, dilation=2)),
        # ("convDS_3x3_SiLU_1dil", depthwise_separable_conv(C_in, C_out, kernel_size=3, padding=2, dilation=2, activation=nn.SiLU())),
        # ("convDS_3x3_Sigmoid_1dil", depthwise_separable_conv(C_in, C_out, kernel_size=3, padding=2, dilation=2, activation=nn.Sigmoid())),
        # ("convDS_3x3_Mish_1dil", depthwise_separable_conv(C_in, C_out, kernel_size=3, padding=2, dilation=2, activation=nn.Mish())),

        ("convDS_5x5_Relu", depthwise_separable_conv(C_in, C_out, kernel_size=5, padding=2)),
        # ("convDS_5x5_SiLU", depthwise_separable_conv(C_in, C_out, kernel_size=5, padding=2, activation=nn.SiLU())),
        # ("convDS_5x5_Sigmoid", depthwise_separable_conv(C_in, C_out, kernel_size=5, padding=2, activation=nn.Sigmoid())),
        # ("convDS_5x5_Mish", depthwise_separable_conv(C_in, C_out, kernel_size=5, padding=2, activation=nn.Mish())),

        # ("convDS_5x5_Relu_1dil", depthwise_separable_conv(C_in, C_out, kernel_size=5, padding=4, dilation=2, activation=nn.SiLU())),
        # ("convDS_5x5_SiLU_1dil", depthwise_separable_conv(C_in, C_out, kernel_size=5, padding=4, dilation=2, activation=nn.SiLU())),
        # ("convDS_5x5_Sigmoid_1dil", depthwise_separable_conv(C_in, C_out, kernel_size=5, padding=4, dilation=2, activation=nn.Sigmoid())),
        # ("convDS_5x5_Mish_1dil", depthwise_separable_conv(C_in, C_out, kernel_size=5, padding=4, dilation=2, activation=nn.Mish())),

        # ("convDS_5x5_Relu_2dil", depthwise_separable_conv(C_in, C_out, kernel_size=5, padding=8, dilation=4, activation=nn.SiLU())),
        # ("convDS_5x5_SiLU_2dil", depthwise_separable_conv(C_in, C_out, kernel_size=5, padding=8, dilation=4, activation=nn.SiLU())),
        # ("convDS_5x5_Sigmoid_2dil", depthwise_separable_conv(C_in, C_out, kernel_size=5, padding=8, dilation=4, activation=nn.Sigmoid())),
        # ("convDS_5x5_Mish_2dil", depthwise_separable_conv(C_in, C_out, kernel_size=5, padding=8, dilation=4, activation=nn.Mish())),

        # ("convDS_7x7_Relu", depthwise_separable_conv(C_in, C_out, kernel_size=7, padding=3)),
        # ("convDS_7x7_SiLU", depthwise_separable_conv(C_in, C_out, kernel_size=7, padding=3, activation=nn.SiLU())),
        # ("convDS_7x7_Sigmoid", depthwise_separable_conv(C_in, C_out, kernel_size=7, padding=3, activation=nn.Sigmoid())),
        # ("convDS_7x7_Mish", depthwise_separable_conv(C_in, C_out, kernel_size=7, padding=3, activation=nn.Mish())),

        # ("convDS_7x7_Relu_1dil", depthwise_separable_conv(C_in, C_out, kernel_size=7, padding=6, dilation=2, activation=nn.SiLU())),
        # ("convDS_7x7_SiLU_1dil", depthwise_separable_conv(C_in, C_out, kernel_size=7, padding=6, dilation=2, activation=nn.SiLU())),
        # ("convDS_7x7_Sigmoid_1dil", depthwise_separable_conv(C_in, C_out, kernel_size=7, padding=6, dilation=2, activation=nn.Sigmoid())),
        # ("convDS_7x7_Mish_1dil", depthwise_separable_conv(C_in, C_out, kernel_size=7, padding=6, dilation=2, activation=nn.Mish())),

        # ("convDS_9x9_Relu", depthwise_separable_conv(C_in, C_out, kernel_size=9, padding=4)),
        # ("convDS_9x9_SiLU", depthwise_separable_conv(C_in, C_out, kernel_size=9, padding=4, activation=nn.SiLU())),
        # ("convDS_9x9_Sigmoid", depthwise_separable_conv(C_in, C_out, kernel_size=9, padding=4, activation=nn.Sigmoid())),
        # ("convDS_9x9_Mish", depthwise_separable_conv(C_in, C_out, kernel_size=9, padding=4, activation=nn.Mish())),

        # ("convDS_9x9_Relu_1dil", depthwise_separable_conv(C_in, C_out, kernel_size=9, padding=8, dilation=2, activation=nn.SiLU())),
        # ("convDS_9x9_SiLU_1dil", depthwise_separable_conv(C_in, C_out, kernel_size=9, padding=8, dilation=2, activation=nn.SiLU())),
        # ("convDS_9x9_Sigmoid_1dil", depthwise_separable_conv(C_in, C_out, kernel_size=9, padding=8, dilation=2, activation=nn.Sigmoid())),
        # ("convDS_9x9_Mish_1dil", depthwise_separable_conv(C_in, C_out, kernel_size=9, padding=8, dilation=2, activation=nn.Mish())),

        # ("convDS_11x11_Relu", depthwise_separable_conv(C_in, C_out, kernel_size=11, padding=5)),
        # ("convDS_11x11_SiLU", depthwise_separable_conv(C_in, C_out, kernel_size=11, padding=5, activation=nn.SiLU())),
        # ("convDS_11x11_Sigmoid", depthwise_separable_conv(C_in, C_out, kernel_size=11, padding=5, activation=nn.Sigmoid())),
        # ("convDS_11x11_Mish", depthwise_separable_conv(C_in, C_out, kernel_size=11, padding=5, activation=nn.Mish())),

        # ("convDS_11x11_Relu_1dil", depthwise_separable_conv(C_in, C_out, kernel_size=11, padding=10, dilation=2, activation=nn.SiLU())),
        # ("convDS_11x11_SiLU_1dil", depthwise_separable_conv(C_in, C_out, kernel_size=11, padding=10, dilation=2, activation=nn.SiLU())),
        # ("convDS_11x11_Sigmoid_1dil", depthwise_separable_conv(C_in, C_out, kernel_size=11, padding=10, dilation=2, activation=nn.Sigmoid())),
        # ("convDS_11x11_Mish_1dil", depthwise_separable_conv(C_in, C_out, kernel_size=11, padding=10, dilation=2, activation=nn.Mish())),
    ])
    return conv_dict



@trace
@model_wrapper
class NodeSpace(nn.Module):
    def __init__(
            self, 
            C_in=1, 
            C_out=1, 
            depth=2, 
            nodes_per_layer=1,
            ops_per_node=1,
            poolOps_per_node=1,
            upsampleOps_per_node=1,
            
            ):
        super().__init__()

        self.depth = depth
        self.nodes = nodes_per_layer
        
        nodes = nodes_per_layer
        start_filters = end_filters = 64
        self.in_layer = nn.Conv2d(C_in, start_filters, kernel_size=3, padding=1)

        # encoder layers
        mid_in = 64
        self.pools = nn.ModuleList()
        self.encoders = nn.ModuleList()
        self.postencoders = nn.ModuleList()
        self.enAttentions = nn.ModuleList()
        for i in range(self.depth):
            self.pools.append(Cell(
                op_candidates=pools(),
                num_nodes=1, 
                num_ops_per_node=poolOps_per_node,
                num_predecessors=1, 
                label=f"pool {i}"
            ))
            self.encoders.append(Cell(
                op_candidates=convs(mid_in,mid_in),
                num_nodes=nodes, 
                num_ops_per_node=ops_per_node,
                num_predecessors=1, 
                label=f"encoder {i}"
            ))
            self.postencoders.append(nn.Conv2d(mid_in*nodes, mid_in*2, kernel_size=3, padding=1)) 
            self.enAttentions.append(attention(mid_in*2,16))
            mid_in *= 2

        # decoder layers
        self.upsamples = nn.ModuleList()
        self.predecoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.postdecoders = nn.ModuleList()
        self.decAttentions = nn.ModuleList()
        for i in range(self.depth):
            self.upsamples.append(Cell(
                op_candidates=upsamples(mid_in,mid_in),
                num_nodes=1, 
                num_ops_per_node=upsampleOps_per_node,
                num_predecessors=1, 
                label=f"upsample {self.depth-i-1}"
            ))
            mid_in //= 2
            self.predecoders.append(nn.Conv2d(mid_in*3, mid_in, kernel_size=3, padding=1))
            self.decoders.append(Cell(
                op_candidates=convs(mid_in,mid_in),
                num_nodes=nodes, 
                num_ops_per_node=ops_per_node,
                num_predecessors=1, 
                label=f"decoder {self.depth-i-1}"
            ))
            self.postdecoders.append(nn.Conv2d(mid_in*nodes, mid_in, kernel_size=3, padding=1))
            self.decAttentions.append(attention(mid_in,16))

        self.out_layer = nn.Conv2d(end_filters, C_out, kernel_size=3, padding=1)
        
    def forward(self, x):
        x = self.in_layer(x)

        skip_connections = [x]

        for i in range(self.depth):
            x = self.pools[i]([x])
            x = self.encoders[i]([x])
            x = self.postencoders[i](x)

            # # attention start
            # b, c, _, _ = x.size()
            # y = nn.AdaptiveAvgPool2d(1)(x).view(b, c)
            # y = self.enAttentions[i](y).view(b, c, 1, 1)
            # x = x * y.expand_as(x)
            x = self.attention_forward(x, self.enAttentions[i])
            # # attention end

            skip_connections.append(x)

        for i in range(self.depth):
            upsampled = self.upsamples[i]([x])
            cropped = self.crop_tensor(upsampled, skip_connections[-(i+2)])
            x = torch.cat([cropped, upsampled], 1)
            x = self.predecoders[i](x)
            x = self.decoders[i]([x])
            x = self.postdecoders[i](x)

            # # attention start
            # b, c, _, _ = x.size()
            # y = nn.AdaptiveAvgPool2d(1)(x).view(b, c)
            # y = self.decAttentions[i](y).view(b, c, 1, 1)
            # x = x * y.expand_as(x)
            x = self.attention_forward(x, self.decAttentions[i])
            # # attention end

        x = self.out_layer(x)
        return x

    def crop_tensor(self, target_tensor, tensor):
        target_size = target_tensor.size()[2]  # Assuming height and width are same
        tensor_size = tensor.size()[2]
        delta = tensor_size - target_size
        delta = delta // 2
        return tensor[:, :, delta:tensor_size-delta, delta:tensor_size-delta]
    
    def attention_forward(self, x, fcs):
        b, c, _, _ = x.size()
        y = nn.AdaptiveAvgPool2d(1)(x).view(b, c)
        y = fcs(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class exportedModel(nn.Module):
    def __init__(self, C_in, C_out, depth, exported_arch):
        super().__init__()

        self.depth = depth
        start_filters = end_filters = 64
        self.in_layer = nn.Conv2d(C_in, start_filters, kernel_size=3, padding=1)

        mid_in = 64
        self.pools = nn.ModuleList()
        self.encoders = nn.ModuleList()
        self.postencoders = nn.ModuleList()
        self.enAttentions = nn.ModuleList()

        for i in range(depth):
            self.pools.append(pools()[exported_arch[f'pool {i}/op_1_0']])
            self.encoders.append(convs(mid_in, mid_in)[exported_arch[f'encoder {i}/op_1_0']])
            self.postencoders.append(nn.Conv2d(mid_in, mid_in*2, kernel_size=3, padding=1)) 
            self.encoders.append(nn.Conv2d(mid_in*2, mid_in*2, kernel_size=3, padding=1))
            self.enAttentions.append(attention(mid_in*2,16))
            mid_in *= 2

        
        # decoder layers
        self.upsamples = nn.ModuleList()
        self.predecoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.postdecoders = nn.ModuleList()
        self.decAttentions = nn.ModuleList()
        for i in range(self.depth):
            self.upsamples.append(upsamples(mid_in, mid_in)[exported_arch[f'upsample {i}/op_1_0']])
            mid_in //= 2
            self.predecoders.append(nn.Conv2d(mid_in*3, mid_in, kernel_size=3, padding=1))
            self.decoders.append(convs(mid_in, mid_in)[exported_arch[f'decoder {i}/op_1_0']])
            self.postdecoders.append(nn.Conv2d(mid_in, mid_in, kernel_size=3, padding=1))
            self.decAttentions.append(attention(mid_in,16))

        self.out_layer = nn.Conv2d(end_filters, C_out, kernel_size=3, padding=1)
        
    def forward(self, x):
        x = self.in_layer(x)
        skip_connections = [x]

        for i in range(self.depth):
            x = self.pools[i](x)
            x = self.encoders[i](x)
            x = self.postencoders[i](x)
            x = self.attention_forward(x, self.enAttentions[i])
            skip_connections.append(x)

        for i in range(self.depth):
            upsampled = self.upsamples[i](x)
            cropped = self.crop_tensor(upsampled, skip_connections[-(i+2)])
            x = torch.cat([cropped, upsampled], 1)
            x = self.predecoders[i](x)
            x = self.decoders[i](x)
            x = self.postdecoders[i](x)
            x = self.attention_forward(x, self.decAttentions[i])


        x = self.out_layer(x)
        return x

    def crop_tensor(self, target_tensor, tensor):
        target_size = target_tensor.size()[2]  # Assuming height and width are same
        tensor_size = tensor.size()[2]
        delta = tensor_size - target_size
        delta = delta // 2
        return tensor[:, :, delta:tensor_size-delta, delta:tensor_size-delta]
    
    def attention_forward(self, x, fcs):
        b, c, _, _ = x.size()
        y = nn.AdaptiveAvgPool2d(1)(x).view(b, c)
        y = fcs(y).view(b, c, 1, 1)
        return x * y.expand_as(x)







# @trace
# class ChannelAttention(nn.Module):
#     def __init__(self, channel, reduction=16, symbol=None, *args, **kwargs):
#         super().__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Sequential(
#             nn.Linear(channel, channel // reduction, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Linear(channel // reduction, channel, bias=False),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         b, c, _, _ = x.size()
#         y = self.avg_pool(x).view(b, c)
#         y = self.fc(y).view(b, c, 1, 1)
#         return x * y.expand_as(x)

    
# @trace
# def conv_2d_with_attention(C_in, C_out, kernel_size=3, dilation=1, padding=1, activation=None):
#     return nn.Sequential(
#         nn.Conv2d(C_in, C_out, kernel_size=kernel_size, dilation=dilation, padding=padding, bias=False),
#         nn.BatchNorm2d(C_out),
#         ChannelAttention(C_out),
#         nn.ReLU() if activation is None else activation,
#         nn.Conv2d(C_out, C_out, kernel_size=kernel_size, dilation=dilation, padding=padding, bias=False),
#         nn.BatchNorm2d(C_out),
#         ChannelAttention(C_out),
#         nn.ReLU() if activation is None else activation
#     )