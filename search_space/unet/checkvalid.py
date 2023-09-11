
import torch
import nni.retiarii.nn.pytorch as nn
from components import convs, pools, upsamples
import itertools

class CheckValidSearchSpace():
    def __init__(self, exported_arch, C_in=1, C_out=1, depth=1, init_features=64):
        super().__init__()

        self.depth = depth
        features = init_features
        self.pools = nn.ModuleList()
        self.encoders = nn.ModuleList()
        self.encoders.append(convs(C_in, features)[exported_arch[f'encoder {1}']])
        self.pools.append(pools()[exported_arch[f'pool {1}']])

        for i in range(depth-1):
            self.encoders.append(convs(features, features * 2)[exported_arch[f'encoder {i+2}']])
            self.pools.append(pools()[exported_arch[f'pool {i+2}']])
            features *= 2

        self.bottleneck = (convs(features, features * 2)[exported_arch[f'bottleneck']])
                           
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(depth):
            self.upconvs.append(upsamples(features * 2, features)[exported_arch[f'upsample {i+1}']])
            self.decoders.append(convs(features * 2, features)[exported_arch[f'decoder {i+1}']])
            features //= 2        

        self.conv = nn.Conv2d(in_channels=features*2, out_channels=C_out, kernel_size=1)

    def forward(self, x):
        skips = []
        
        # Loop through encoders and pools
        for i in range(self.depth):
            try:
                x = self.encoders[i](x)
            except Exception as e:
                current_choice = self.extract_layers_info(self.encoders[i].__class__.__name__)
                previous_choice = self.extract_layers_info(self.encoders[i-1].__class__.__name__) if i != 0 else "None"
                raise RuntimeError(f"Failed at encoder {i+1} due to {current_choice} (previous: {previous_choice}): " + str(e))
            
            skips.append(x)
            
            try:
                x = self.pools[i](x)
            except Exception as e:
                current_choice = self.extract_layers_info(self.pools[i].__class__.__name__)
                previous_choice = self.extract_layers_info(self.encoders[i].__class__.__name__)
                raise RuntimeError(f"Failed at pool {i+1} due to {current_choice} (previous: {previous_choice}): " + str(e))
        
        # Bottleneck
        try:
            x = self.bottleneck(x)
        except Exception as e:
            current_choice = self.extract_layers_info(self.bottleneck.__class__.__name__)
            previous_choice = self.extract_layers_info(self.pools[-1].__class__.__name__)
            raise RuntimeError(f"Failed at bottleneck due to {current_choice} (previous: {previous_choice}): " + str(e))
        
        # Loop through upconvs and decoders
        for i in range(self.depth):
            try:
                x = self.upconvs[i](x)
            except Exception as e:
                current_choice = self.extract_layers_info(self.upconvs[i].__class__.__name__)
                previous_choice = self.extract_layers_info(self.decoders[i-1].__class__.__name__) if i != 0 else self.extract_layers_info(self.bottleneck.__class__.__name__)
                raise RuntimeError(f"Failed at upconv {i+1} due to {current_choice} (previous: {previous_choice}): " + str(e))
            
            try:
                x = torch.cat((x, skips.pop()), dim=1)
            except Exception as e:
                current_choice = "torch.cat"
                previous_choice = self.extract_layers_info(self.upconvs[i].__class__.__name__)
                raise RuntimeError(f"Failed at concatenation {i+1} due to {current_choice} (previous: {previous_choice}): " + str(e))

            try:
                x = self.decoders[i](x)
            except Exception as e:
                current_choice = self.extract_layers_info(self.decoders[i].__class__.__name__)
                previous_choice = self.extract_layers_info(self.upconvs[i].__class__.__name__)
                raise RuntimeError(f"Failed at decoder {i+1} due to {current_choice} (previous: {previous_choice}): " + str(e))

        return torch.sigmoid(self.conv(x))

    def extract_layers_info(self, module):
        """Extract detailed info about layers from a given module."""
        if isinstance(module, nn.Sequential):
            # Joining the names of the contained modules to get a string representation
            return "_".join([layer.__class__.__name__ for layer in module.children()])
        else:
            return module.__class__.__name__

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
        x = torch.randn(1, 1, 64, 64)
        
        try:
            out = self.forward(x)
            assert out.shape == (1, 1, 64, 64)
            print('Test passed')
        except RuntimeError as e:
            print(str(e))
            raise e



def generate_list():
    # List of functions to get candidates for each layer
    funcs = [
        lambda: convs(1,64), 
        pools, 
        lambda: convs(64,128), 
        lambda: upsamples(128,64), 
        lambda: convs(128,64)
    ]

    # Generate the Cartesian product of all candidates for each layer
    all_combinations = list(itertools.product(*[func() for func in funcs]))

    # Define the filename for clarity and potential future changes
    filename = "failed_architectures.txt"

    # Clear the content of the file before starting the loop (if the file exists)
    with open(filename, "w") as file:
        pass

    for combination in all_combinations:
        exported_arch = {
            "encoder 1": combination[0],
            "pool 1": combination[1],
            "bottleneck": combination[2],
            "upsample 1": combination[3],
            "decoder 1": combination[4]
        }

        checker = CheckValidSearchSpace(exported_arch)

        try:
            checker.test()
        except RuntimeError as e:
            # Append the failure reason (e.g., "Failed at encoder 1") to the file
            with open(filename, "a") as file:
                file.write(str(e) + "\n")
