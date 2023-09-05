import nni
from nni.retiarii.evaluator.pytorch import Lightning, Trainer
from nni.retiarii.evaluator.pytorch.lightning import DataLoader

import numpy as np
import torch

# import sys to import from different directory
import sys
sys.path.insert(1, '/home/joe/nas-for-dip/')
from search_eval.utils.common_utils import *
from search_eval.eval_no_search_SGLD_ES import Eval_SGLD_ES, SingleImageDataset

# make sure cuda is available and ready to go
torch.cuda.empty_cache()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
print('CUDA available: {}'.format(torch.cuda.is_available()))

params = {
    'learning_rate': 0.14,
    'buffer_size': 700,
    'patience': 200,
    'weight_decay': 1.5e-7,
}

optimized_params = nni.get_next_parameter()
params.update(optimized_params)
print(params)

# INPUTS
total_iterations = 1400
show_every = 10
resolution = 64
noise_level = '0.09'
noise_type = 'gaussian'
phantom = np.load(f'/home/joe/nas-for-dip/phantoms/ground_truth/{resolution}/{45}.npy')
phantom_noisy = np.load(f'/home/joe/nas-for-dip/phantoms/{noise_type}/res_{resolution}/nl_{noise_level}/p_{45}.npy')

model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet', in_channels=1, out_channels=1, init_features=64, pretrained=False)

print(f"\n\n----------------------------------")
print(f'Experiment Configuration:')

print(f'\tTotal Iterations: {total_iterations}')
print(f'\tPatience: {params["patience"]}')
print(f'\tBuffer Size: {params["buffer_size"]}')
print(f'\tLearning Rate: {params["learning_rate"]}')
print(f'\tWeight Decay: {params["weight_decay"]}')
print(f'\tImage Resolution: {resolution}')

print(f'\tPlotting every {show_every} iterations')
print(f"----------------------------------\n\n")

# Create the lightning module
module = Eval_SGLD_ES(
                phantom=phantom, 
                phantom_noisy=phantom_noisy,

                learning_rate=params['learning_rate'], 
                patience=params['patience'],
                buffer_size=params['buffer_size'],
                weight_decay=params['weight_decay'],
                
                model=model, # model defaults to U-net 
                show_every=show_every,
                )

# Create a PyTorch Lightning trainer
trainer = Trainer(
            max_epochs=total_iterations,
            fast_dev_run=False,
            gpus=1,
            )
            
if not hasattr(trainer, 'optimizer_frequencies'):
    trainer.optimizer_frequencies = []

# Create the lighting object for evaluator
train_loader = DataLoader(SingleImageDataset(phantom, num_iter=1), batch_size=1)

lightning = Lightning(lightning_module=module, trainer=trainer, train_dataloaders=train_loader, val_dataloaders=None)
lightning.fit(model)