
from collections import OrderedDict
import torch
import nni.retiarii.nn.pytorch as nn
from nni import trace
from nni.retiarii import model_wrapper
from nni.retiarii.nn.pytorch import Cell
from .utils.components import pools, upsamples, convs
from .utils.attention import *
@trace
@model_wrapper
class UNetSpace(nn.Module):
    def __init__(
            self, 
            C_in=1, 
            C_out=1, 
            depth=2, 
            init_features=64,
            nodes_per_layer=2, # accept only 1 or 2,
            ops_per_node=1,
            use_attention=False
            ):
        super().__init__()

        self.depth = depth
        self.nodes = nodes_per_layer
        ennodes = nodes_per_layer
        denodes = 1
        self.use_attention = use_attention
        filters = 64
        filters_start = filters

        self.enConvList = nn.ModuleList()
        self.decConvList = nn.ModuleList()
        self.upList = nn.ModuleList()
        self.poolList = nn.ModuleList()
        self.enAttentions = nn.ModuleList()

        ### initialize Encoders
        self.enConvList.append(
            Cell(
                op_candidates=convs(C_in,filters),
                num_nodes=1,
                num_ops_per_node=1,
                num_predecessors=1,
                label=f"encoder 1"
                ))
        self.poolList.append(
            Cell(
                op_candidates=pools(),
                num_nodes=1,
                num_ops_per_node=1,
                num_predecessors=1,
                label=f"pool 1"
            )
        )
        self.enAttentions.append(seBlock(filters,16))

        for i in range(depth-1):
            self.poolList.append(
                Cell(
                    op_candidates=pools(),
                    num_nodes=1,
                    num_ops_per_node=1,
                    num_predecessors=1,
                    label=f"pool {i+2}"
                ))
            self.enConvList.append(
                Cell(
                    op_candidates=convs(filters,filters*2//ennodes),
                    num_nodes=ennodes,
                    num_ops_per_node=1,
                    num_predecessors=1,
                    label=f"encoder {i+2}"
                    ))
            self.enAttentions.append(seBlock(filters*2,16))
            
            filters *= 2

        ### initialize Bottleneck
        self.bottleneck = Cell(
                        op_candidates=convs(filters,filters*2//ennodes),
                        num_nodes=ennodes,
                        num_ops_per_node=1,
                        num_predecessors=1,
                        label=f"bottleneck"
                        )
        self.enAttentions.append(seBlock(filters*2,16))

        ### initialize Decoders
        for i in range(depth):
            self.upList.append(nn.ConvTranspose2d(filters * 2, filters, kernel_size=2, stride=2))
            self.decConvList.append(
                Cell(
                    op_candidates=convs(filters*2,filters//denodes),
                    num_nodes=denodes,
                    num_ops_per_node=1,
                    num_predecessors=1,
                    label=f"decoder {i+1}"
                    ))
            filters //= 2

        self.outconv = nn.Conv2d(in_channels=filters_start, out_channels=C_out, kernel_size=1)
        
    def forward(self, x):
        skips = []
        for enconv, pl, att in zip(self.enConvList, self.poolList, self.enAttentions):
            x = enconv([x])
            if self.use_attention:
                x = seForward(x, att)
            skips.append(x)
            x = pl(x)

        x = self.bottleneck([x])
        if self.use_attention:
            x = seForward(x, self.enAttentions[-1])

        skips = skips[::-1]
        for i, deconv, ups in zip(range(self.depth), self.decConvList, self.upList):
            x = ups(x)
            x = torch.cat((x, skips[i]), dim=1)
            x = deconv([x])
        return torch.sigmoid(self.outconv(x))
    
    def attention_forward(self, x, fcs):
        b, c, _, _ = x.size()
        y = nn.AdaptiveAvgPool2d(1)(x).view(b, c)
        y = fcs(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

    def register_hooks(self):
        # Function to print the shape of the tensor after forward pass of each module
        def print_shape(name):
            def hook(module, input, output):
                print(f"{name} shape: {output.shape}")
            return hook

        for enConv, decConv, up in zip(self.enConvList, self.decConvList, self.upList):
            enConv.register_forward_hook(print_shape("enConv"))
            decConv.register_forward_hook(print_shape("decConv"))
            up.register_forward_hook(print_shape("up"))
        self.bottleneck.register_forward_hook(print_shape("bottleneck"))

    def test(self):
        # Register hooks
        self.register_hooks()

        x = torch.randn(1, 1, 64, 64)
        print(f"input shape: {x.shape}")
        out = self.forward(x)
        print(out.shape)
        assert out.shape == (1, 1, 64, 64)
        print('Test passed')

class exportedUNet(nn.Module):
    def __init__(self, exported_arch, depth, C_in=1, C_out=1):
        super().__init__()

        self.depth = depth
        start_filters = end_filters = 64
        self.in_layer = nn.Conv2d(C_in, start_filters, kernel_size=3, padding=1)

        mid_in = 64
        self.enConvList = nn.ModuleList()
        self.decConvList = nn.ModuleList()
        self.upList = nn.ModuleList()
        self.poolList = nn.ModuleList()

        for i in range(depth):
            self.poolList.append(pools()[exported_arch[f'pool {i+1}/op_1_0']])
            self.enConvList.append(convs(mid_in, mid_in)[exported_arch[f'encoder {i+1}/op_1_0']])
            mid_in *= 2

        # bottleneck
        self.bottleneck = convs(mid_in, mid_in*2)[exported_arch[f'bottleneck/op_1_0']]
        
        # decoder layers
        for i in range(self.depth):
            self.upList.append(upsamples(mid_in, mid_in)[exported_arch[f'upsample {i+1}/op_1_0']])
            self.decConvList.append(convs(mid_in, mid_in)[exported_arch[f'decoder {i+1}/op_1_0']])
            mid_in //= 2

        self.out_layer = nn.Conv2d(end_filters, C_out, kernel_size=3, padding=1)
        
    def forward(self, x):

        skips = []
        for pl, encov in zip(self.poolList, self.enConvList):
            x = encov(x)
            skips.append(x)
            x = pl(x)

        x = self.bottleneck(x)

        skips = skips[::-1]
        for i, up, decov in zip(range(self.depth), self.upList, self.decConvList):
            x = up(x)
            x = torch.cat((x, skips[i]), dim=1)
            x = decov(x)

        x = self.out_layer(x)
        return torch.sigmoid(x)
    
    def attention_forward(self, x, fcs):
        b, c, _, _ = x.size()
        y = nn.AdaptiveAvgPool2d(1)(x).view(b, c)
        y = fcs(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
    
    def register_hooks(self):
        # Function to print the shape of the tensor after forward pass of each module
        def print_shape(name):
            def hook(module, input, output):
                print(f"{name} shape: {output.shape}")
            return hook

        for enConv, decConv, up in zip(self.enConvList, self.decConvList, self.upList):
            enConv.register_forward_hook(print_shape("enConv"))
            decConv.register_forward_hook(print_shape("decConv"))
            up.register_forward_hook(print_shape("up"))
        self.bottleneck.register_forward_hook(print_shape("bottleneck"))

    def test(self):
        # Register hooks
        self.register_hooks()

        x = torch.randn(1, 1, 64, 64)
        print(f"input shape: {x.shape}")
        out = self.forward(x)
        print(out.shape)
        assert out.shape == (1, 1, 64, 64)
        print('Test passed')