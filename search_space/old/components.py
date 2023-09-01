
import torch
import nni.retiarii.nn.pytorch as nn

from collections import OrderedDict

class Convolutions(nn.Module):
    def __init__(self, out_channels, activations, convs1, convs2, layer_name):
        super().__init__()

        self.conv1 = nn.LayerChoice(convs1, label=f'{layer_name} - Step 1: Convolution 1')
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.LayerChoice(activations, label=f'{layer_name} - Step 2: Activation 1')
        
        self.conv2 = nn.LayerChoice(convs2, label=f'{layer_name} - Step 3: Convolution 2')
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.LayerChoice(activations, label=f'{layer_name} - Step 4: Activation 2')

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        return x
    
class BaseBlock(nn.Module):
    def __init__(self):
        super(BaseBlock, self).__init__()

    def get_conv_ordered_dict(self, in_channels, out_channels, ks, pd, dl, first=True):
        layers = [
            ("Conv2d", nn.Conv2d(in_channels, out_channels, kernel_size=ks, padding=pd, dilation=dl)),
            ("DepthwiseSeparable", nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=ks, padding=pd, dilation=dl, groups=in_channels),
                nn.Conv2d(in_channels, out_channels, kernel_size=1)
                )
            )
        ]
        if not first:
            layers.append(("Depthwise", nn.Conv2d(in_channels, out_channels, kernel_size=1)))
        return OrderedDict(layers)

    def crop_tensor(self, target_tensor, tensor):
        target_size = target_tensor.size()[2]  # Assuming height and width are same
        tensor_size = tensor.size()[2]
        delta = tensor_size - target_size
        delta = delta // 2
        return tensor[:, :, delta:tensor_size-delta, delta:tensor_size-delta]

class EncoderBlock(BaseBlock):
    def __init__(self, in_channels, out_channels, ks, pd, dl, activations, downsamples, layer_name):
        super(EncoderBlock, self).__init__()
        
        self.downsample = nn.LayerChoice(downsamples,label=f'{layer_name} - Step 0: Downsampling Technique')
        self.conv_layer = Convolutions(out_channels, 
                                       activations, 
                                       self.get_conv_ordered_dict(in_channels, out_channels, ks, pd, dl),
                                       self.get_conv_ordered_dict(out_channels, out_channels, ks, pd, dl, first=False), 
                                       layer_name)

    def forward(self, x):
        x = self.downsample(x)
        x = self.conv_layer(x)
        return x


class DecoderBlock(BaseBlock):
    def __init__(self, in_channels, out_channels, ks, pd, dl, activations, upsamples, layer_name):
        super(DecoderBlock, self).__init__()

        self.upsample = nn.LayerChoice(upsamples, label=f"{layer_name} - Step 0: Upsampling Technique")
        self.conv_layer = Convolutions(out_channels, 
                                       activations, 
                                       self.get_conv_ordered_dict(in_channels, out_channels, ks, pd, dl),
                                       self.get_conv_ordered_dict(out_channels, out_channels, ks, pd, dl, first=False), 
                                       layer_name)

    def forward(self, x, skip):
        upsampled = self.upsample(x)
        cropped = self.crop_tensor(upsampled, skip)
        return self.conv_layer(torch.cat([cropped, upsampled], 1))