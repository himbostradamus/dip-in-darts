from collections import OrderedDict
import torch
import nni.retiarii.nn.pytorch as nn
from .utils.attention import seBlock, seForward

# this is the original UNet
class UNet(torch.nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=32, depth=4, use_attention=True):
        super(UNet, self).__init__()

        self.depth = depth
        self.use_attention = use_attention

        features = init_features
        self.pools = nn.ModuleList()
        self.encoders = nn.ModuleList()
        self.enAttentions = nn.ModuleList()

        self.encoders.append(UNet._block(in_channels, features, name="enc1"))
        self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2))

        for i in range(depth-1):
            self.encoders.append(UNet._block(features, features * 2, name=f"enc{i+2}"))
            self.enAttentions.append(seBlock(features*2,16))
            self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
            features *= 2

        self.bottleneck = UNet._block(features, features * 2, name="bottleneck")
        self.enAttentions.append(seBlock(features*2,16))

        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(depth):
            self.upconvs.append(nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2))
            self.decoders.append(UNet._block(features * 2, features, name=f"dec{i+1}"))
            features //= 2
        
        self.conv = nn.Conv2d(in_channels=features*2, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        skips = []
        for i in range(self.depth):
            x = self.encoders[i](x)
            if self.use_attention:
                x = seForward(x, self.enAttentions[i])
            skips.append(x)
            x = self.pools[i](x)
            
        x = self.bottleneck(x)
        if self.use_attention:
            x = seForward(x, self.enAttentions[-1])

        for i in range(self.depth):
            x = self.upconvs[i](x)
            x = torch.cat((x, skips[-i-1]), dim=1)
            x = self.decoders[i](x)

        return torch.sigmoid(self.conv(x))

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )

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