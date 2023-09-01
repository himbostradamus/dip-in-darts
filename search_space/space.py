import logging
import nni.retiarii.nn.pytorch as nn

from .components import EncoderBlock, DecoderBlock, Convolutions, BaseBlock

from collections import OrderedDict
from nni.retiarii import model_wrapper


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@model_wrapper
class SearchSpace(BaseBlock):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()

        network_depth = nn.ValueChoice([1, 2, 3, 4], label="Network Depth")
        # maybe can use max of depth to create a better label down in the decoder_block = ... line below

        ks = nn.ValueChoice([1, 3, 5, 7, 9], label="Kernel Size")
        dl = nn.ValueChoice([1, 2, 3, 4, 5], label="Dilation Rate")
        pd = (ks - 1) * dl // 2

        activations = OrderedDict([
            ("RelU", nn.ReLU(inplace=True)),
            ("LeakyRelU", nn.LeakyReLU(inplace=True)),
            ("Sigmoid", nn.Sigmoid()),
            ("Selu", nn.SELU(inplace=True)),
            ("PreLU", nn.PReLU()),
            ("SiLU", nn.SiLU(inplace=True)),
        ])

        downsamples = OrderedDict([
            ("AvgPool2d", nn.AvgPool2d(kernel_size=2, stride=2)),
            ("MaxPool2d", nn.MaxPool2d(kernel_size=2, stride=2)),
        ])

        upsamples = OrderedDict([
            ("Nearest", nn.Upsample(scale_factor=2,mode='nearest')),
            ("Bilinear", nn.Upsample(scale_factor=2,mode='bilinear', align_corners=True)),
            ("Bicubic", nn.Upsample(scale_factor=2,mode='bicubic', align_corners=True))
        ])

        # Conv layer in"
        self.mid_channels = 64
        self.convs1 = self.get_conv_ordered_dict(in_channels, self.mid_channels, ks, pd, dl)
        self.convs2 = self.get_conv_ordered_dict(self.mid_channels, self.mid_channels, ks, pd, dl, first=False)
        self.first = Convolutions(self.mid_channels, activations, self.convs1, self.convs2, "First Conv Layer")


        # For Encoders:
        encoder_block = lambda index: EncoderBlock(64*(2**index), 64*(2**(index+1)), ks, pd, dl, activations, downsamples, f"Encoder {index+1}")
        self.encoders = nn.Repeat(encoder_block, network_depth)

        # For Decoders:
        decoder_block = lambda index: DecoderBlock(64*(2**(index))*3, 64*(2**index), ks, pd, dl, activations, upsamples, f"Decoder {index+1}")
        self.decoders = nn.Repeat(decoder_block, network_depth)
        self.decoders = self.decoders[::-1]

        # Conv layer out
        self.out = nn.Conv2d(64, out_channels, kernel_size=ks, padding=pd, dilation=dl)
        
    def forward(self, x):
        logger.info("Input: %s", x.size())
        
        # Variables to store intermediate values
        encoder_outputs = []

        # Start with the first conv layer
        x = self.first(x)
        encoder_outputs.append(x)
        logger.info(f"Initial Conv Layer: %s", x.size())

        # Encoder pass
        for i, encoder in enumerate(self.encoders):
            x = encoder(x)
            encoder_outputs.append(x)
            logger.info(f"Encoder {i+1}: %s", x.size())

        # Decoder pass
        for i, decoder in enumerate(self.decoders):
            x = decoder(x, encoder_outputs[-(i+2)])
            logger.info(f"Decoder {len(self.decoders) - i}: %s", x.size())

        x = self.out(x)
        logger.info("Output: %s", x.size())
        return x
    
    def get_conv_ordered_dict(self, in_channels, out_channels, ks, pd, dl, first=False):
         layers = [
                ("Conv2d", nn.Conv2d(in_channels, out_channels, kernel_size=ks, padding=pd, dilation=dl)),
                ("DepthwiseSeparable", nn.Sequential(
                    nn.Conv2d(in_channels, in_channels, kernel_size=ks, padding=pd, dilation=dl, groups=in_channels),
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, dilation=1)
                    )
                )
            ]
         if not first:
            layers.append(("Depthwise", nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, dilation=1)))
         return OrderedDict(layers)