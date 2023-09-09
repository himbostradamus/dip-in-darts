from collections import OrderedDict
import torch
import torch.nn as nn


def conv_2d(C_in, C_out, kernel_size=3, dilation=1, padding=1, activation=None):
    return nn.Sequential(
        nn.Conv2d(C_in, C_out, kernel_size=kernel_size, dilation=dilation, padding=padding, bias=False),
        nn.BatchNorm2d(C_out),
        nn.ReLU() if activation is None else activation,
        nn.Conv2d(C_out, C_out, kernel_size=kernel_size, dilation=dilation, padding=padding, bias=False),
        nn.BatchNorm2d(C_out),
        nn.ReLU() if activation is None else activation
    )

def transposed_conv_2d(C_in, C_out, kernel_size=4, stride=2, padding=1, activation=None):
    return nn.Sequential(
        nn.ConvTranspose2d(C_in, C_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(C_out),
        nn.ReLU() if activation is None else activation
    )

def convolutions(c_in, c_out, kernel_size=3, padding=1):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=kernel_size, padding=padding),
        nn.BatchNorm2d(c_out),
        nn.ReLU(),
        ChannelAttention(c_out),
        nn.Conv2d(c_out, c_out, kernel_size=kernel_size, padding=padding),
        nn.BatchNorm2d(c_out),
        nn.ReLU(),
        ChannelAttention(c_out),
    )

# class ChannelAttention(nn.Module):
#     def __init__(self, in_planes, ratio=16):
#         super(ChannelAttention, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)

#         self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, kernel_size=1, bias=False)
#         self.relu1 = nn.ReLU()
#         self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, kernel_size=1, bias=False)

#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
#         max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
#         out = avg_out + max_out
#         return self.sigmoid(out)

class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False), 
            nn.Sigmoid()
        )
        
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SillyNet(nn.Module):
    def __init__(
            self, 
            C_in=1, 
            C_out=1, 
            depth=2,             
            ):
        super().__init__()

        self.depth = depth
        mid_in = 64
        self.in_layer = nn.Conv2d(C_in, mid_in, kernel_size=3, padding=1)

        # encoder layers
        self.pools = nn.ModuleList()
        self.encoders = nn.ModuleList()
        for _ in range(self.depth):
            self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
            self.encoders.append(convolutions(mid_in, mid_in*2))
            mid_in *= 2

        # decoder layers
        self.upsamples = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for _ in range(self.depth):
            self.upsamples.append(transposed_conv_2d(mid_in, mid_in, kernel_size=2, stride=2, padding=0))
            mid_in //= 2
            self.decoders.append(convolutions(mid_in*3, mid_in))

        self.out_layer = nn.Conv2d(mid_in, C_out, kernel_size=3, padding=1)
        
    def forward(self, x):
        x = self.in_layer(x)

        skip_connections = [x]

        for i in range(self.depth):
            x = self.pools[i](x)
            x = self.encoders[i](x)
            skip_connections.append(x)

        for i in range(self.depth):
            upsampled = self.upsamples[i](x)
            cropped = self.crop_tensor(upsampled, skip_connections[-(i+2)])
            x = torch.cat([cropped, upsampled], 1)
            x = self.decoders[i](x)

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
    
    def up_crop_and_concat(self, x, skip_connections, i):
        upsampled = self.upsamples[i](x)
        cropped = self.crop_tensor(upsampled, skip_connections[-(i+2)])
        x = torch.cat([cropped, upsampled], 1)
        return x
