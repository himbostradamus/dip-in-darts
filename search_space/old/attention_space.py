    
import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, kernel_size=1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, kernel_size=1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

# class ChannelAttention(nn.Module):
#     def __init__(self, channel, reduction=16):
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

class DeepImagePrior(nn.Module):
    def __init__(self, in_channels, out_channels, depth):
        super(DeepImagePrior, self).__init__()

        mid_depth = depth // 2

        # Input layer
        self.in_layer = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.convs_enc = nn.ModuleList()
        self.cas_enc = nn.ModuleList()

        # Encoding
        for i in range(mid_depth):
            self.convs_enc.append(nn.Conv2d(2**i * 64, 2**(i+1) * 64, kernel_size=3, padding=1))
            self.cas_enc.append(ChannelAttention(2**(i+1) * 64))

        self.convs_dec = nn.ModuleList()
        self.cas_dec = nn.ModuleList()

        # Decoding
        for i in range(mid_depth, depth - 1):
            self.convs_dec.append(nn.Conv2d(2**(depth-i-1) * 64, 2**(depth-i-2) * 64, kernel_size=3, padding=1))
            self.cas_dec.append(ChannelAttention(2**(depth-i-2) * 64))

        # Output layer
        self.out_layer = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.relu(self.in_layer(x))
        
        for conv, ca in zip(self.convs_enc, self.cas_enc):
            x = F.relu(conv(x))
            x = ca(x) * x

        for conv, ca in zip(self.convs_dec, self.cas_dec):
            x = F.relu(conv(x))
            x = ca(x) * x

        x = self.out_layer(x)
        return x