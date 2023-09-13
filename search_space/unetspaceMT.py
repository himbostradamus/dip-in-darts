import torch
import nni.retiarii.nn.pytorch as nn

from nni.retiarii import model_wrapper
from nni.retiarii.nn.pytorch import LayerChoice
from .components import pools, upsamples, convs

# this search space is for multi-trial search strategies
@model_wrapper
class UNetSpaceMT(torch.nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=32, depth=4):
        super().__init__()

        self.depth = depth

        features = init_features

        # encoder layers initialize
        self.pools = nn.ModuleList()
        self.encoders = nn.ModuleList()

        # first encoder layer
        self.encoders.append(LayerChoice(convs(in_channels, features),label='enc1'))
        self.pools.append(LayerChoice(pools(),label='pool1'))

        # remaining encoder layers
        for i in range(depth-1):
            self.encoders.append(LayerChoice(convs(features, features * 2),label=f"enc{i+2}"))
            self.pools.append(LayerChoice(pools(),label=f"pool{i+2}"))
            features *= 2

        # bottleneck layer (bottom of unet)
        self.bottleneck = LayerChoice(convs(features, features * 2),label="bottleneck")

        # decoder layers initialize
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()

        # all decoder layers
        for i in range(depth):
            self.upconvs.append(LayerChoice(upsamples(features * 2, features),label=f"upconv{i+1}"))
            self.decoders.append(LayerChoice(convs(features * 2, features),label=f"dec{i+1}"))
            features //= 2        

        # final conv layer
        self.conv = nn.Conv2d(in_channels=features*2, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        skips = []
        for i in range(self.depth):
            x = self.encoders[i](x)
            skips.append(x)
            x = self.pools[i](x)
            
        x = self.bottleneck(x)

        for i in range(self.depth):
            x = self.upconvs[i](x)
            x = torch.cat((x, skips[-i-1]), dim=1)
            x = self.decoders[i](x)

        return torch.sigmoid(self.conv(x))

    def register_hooks(self):
        # Function to print the shape of the tensor after forward pass of each module
        def print_shape(name):
            def hook(module, input, output):
                print(f"{name} shape: {output.shape}")
            return hook

        # Registering hooks for specific layers to capture intermediate tensor details
        for encoder, pool in zip(self.encoders, self.pools):
            encoder.register_forward_hook(print_shape("encoder"))
            pool.register_forward_hook(print_shape("pool"))

        self.bottleneck.register_forward_hook(print_shape("bottleneck"))
        for upconv, decoder in zip(self.upconvs, self.decoders):
            upconv.register_forward_hook(print_shape("upconv"))
            decoder.register_forward_hook(print_shape("decoder"))


    def test(self):
        # Register hooks
        self.register_hooks()

        x = torch.randn(1, 1, 64, 64)
        print(f"input shape: {x.shape}")
        out = self.forward(x)
        print(out.shape)
        assert out.shape == (1, 1, 64, 64)
        print('Test passed')