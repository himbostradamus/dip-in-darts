from collections import OrderedDict
import torch
from nni import trace
from nni.retiarii import model_wrapper
from nni.retiarii.nn.pytorch import Cell
from .components import conv_2d, depthwise_separable_conv, transposed_conv_2d, attention, pools, upsamples, convs
import math

# choose one
import nni.retiarii.nn.pytorch as nn
# import torch.nn as nn

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