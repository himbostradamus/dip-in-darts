
from collections import OrderedDict
import torch
import nni.retiarii.nn.pytorch as nn
from nni import trace
from nni.retiarii import model_wrapper
from nni.retiarii.nn.pytorch import Cell
from .components import conv_2d, depthwise_separable_conv, transposed_conv_2d, attention, pools, upsamples, convs

@trace
@model_wrapper
class UNetSpace(nn.Module):
    def __init__(
            self, 
            C_in=1, 
            C_out=1, 
            depth=2, 
            nodes_per_layer=1, # accept only 1 or 2,
            ops_per_node=1,
            poolOps_per_node=1,
            upsampleOps_per_node=1,
            use_attention=False
            
            ):
        super().__init__()

        self.depth = depth
        self.nodes = nodes_per_layer
        nodes = nodes_per_layer
        self.use_attention = use_attention
        

        # encoder layers
        end_filters = 64
        mid_in = 64
        self.pools = nn.ModuleList()
        self.preencoders = nn.ModuleList()
        self.encoders = nn.ModuleList()
        self.postencoders = nn.ModuleList()
        self.enAttentions = nn.ModuleList()

        self.preencoders.append(nn.Conv2d(C_in, mid_in, kernel_size=3, padding=1))
        self.encoders.append(Cell(
                op_candidates=convs(mid_in,mid_in),
                num_nodes=1,
                num_ops_per_node=ops_per_node,
                num_predecessors=1,
                label=f"encoder 1"
            ))
        self.enAttentions.append(attention(mid_in,16))
        self.pools.append(Cell(
                op_candidates=pools(),
                num_nodes=1, 
                num_ops_per_node=poolOps_per_node,
                num_predecessors=1, 
                label=f"pool 1"
            ))

        for i in range(self.depth-1):
            self.pools.append(Cell(
                op_candidates=pools(),
                num_nodes=1,
                num_ops_per_node=poolOps_per_node,
                num_predecessors=1,
                label=f"pool {i+2}"
            ))
            
            if nodes == 1:
                self.preencoders.append(nn.Conv2d(mid_in, mid_in*2, kernel_size=3, padding=1))
                encoder_in_channels, encoder_out_channels = mid_in*2, mid_in*2
            else:
                encoder_in_channels, encoder_out_channels = mid_in, mid_in
                
            self.encoders.append(Cell(
                op_candidates=convs(encoder_in_channels, encoder_out_channels),
                num_nodes=nodes,
                num_ops_per_node=ops_per_node,
                num_predecessors=1,
                label=f"encoder {i+2}"
            ))

            self.enAttentions.append(attention(mid_in*2, 16))
            mid_in *= 2


        if nodes == 1:
            self.preencoders.append(nn.Conv2d(mid_in, mid_in*2, kernel_size=3, padding=1))
            encoder_in_channels, encoder_out_channels = mid_in*2, mid_in*2
        else:
            encoder_in_channels, encoder_out_channels = mid_in, mid_in
        self.bottleneck = Cell(
                op_candidates=convs(encoder_in_channels,encoder_out_channels),
                num_nodes=nodes,
                num_ops_per_node=ops_per_node,
                num_predecessors=1,
                label=f"bottleneck"
            )

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
            self.predecoders.append(nn.Conv2d(mid_in*3, mid_in, kernel_size=3, padding=1))
            self.decoders.append(Cell(
                op_candidates=convs(mid_in,mid_in),
                num_nodes=nodes, 
                num_ops_per_node=ops_per_node,
                num_predecessors=1, 
                label=f"decoder {self.depth-i-1}"
            ))
            self.decAttentions.append(attention(mid_in,16))
            mid_in //= 2

        self.out_layer = nn.Conv2d(end_filters, C_out, kernel_size=3, padding=1)
        
    def forward(self, x):
        print(f'input shape: {x.shape}')
        x = self.in_layer(x)
        print(f'after in_layer: {x.shape}')
        skip_connections = [x]

        for i in range(self.depth):
            x = self.pools[i]([x])
            x = self.preencoders[i](x)
            x = self.encoders[i]([x])
            if self.use_attention:
                x = self.attention_forward(x, self.enAttentions[i])
            skip_connections.append(x)

        for i in range(self.depth):
            upsampled = self.upsamples[i]([x])
            cropped = self.crop_tensor(upsampled, skip_connections[-(i+2)])
            x = torch.cat([cropped, upsampled], 1)
            x = self.predecoders[i](x)
            x = self.decoders[i]([x])
            if self.use_attention:
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
    

class exportedUNet(nn.Module):
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
            self.encoders.append(convs(mid_in, mid_in*2)[exported_arch[f'encoder {i}/op_1_0']])
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
            self.decoders.append(convs(mid_in*3, mid_in)[exported_arch[f'decoder {i}/op_1_0']])

        self.out_layer = nn.Conv2d(end_filters, C_out, kernel_size=3, padding=1)
        
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

