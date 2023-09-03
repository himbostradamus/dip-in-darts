
#               ,;;;,
#              ;;;;;;;
#           .-'`\, '/_
#         .'   \ ("`(_)
#        / `-,.'\ \_/
#        \  \/\  `--`
#         \  \ \
#          / /| |
#         /_/ |_|
#        ( _\ ( _\  #:##        #:##        #:##         #:##
#                         #:##        #:##        #:##

import nni
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

params = {
        'learning_rate': 0.01,
        'patience': 200,
        'buffer_size': 100,
        'weight_decay': 5e-8, # this is proportionate to a 1024x1024 image
        }

optimized_params = nni.get_next_parameter()
params.update(optimized_params)
print(params)

# load dataset
