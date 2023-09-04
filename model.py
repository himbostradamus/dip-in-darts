import nni
from nni.retiarii.evaluator.pytorch import Lightning, Trainer
from nni.retiarii.evaluator.pytorch.lightning import DataLoader

import numpy as np
import torch

from search_eval.utils.common_utils import *
from search_eval.eval_no_search_SGLD import Eval_SGLD, SingleImageDataset
from phantoms.noises import add_gaussian_noise

# make sure cuda is available and ready to go
torch.cuda.empty_cache()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
print('CUDA available: {}'.format(torch.cuda.is_available()))

params = {
    'lr': 0.001,
    'burnin_iter': 700,

}

optimized_params = nni.get_next_parameter()
params.update(optimized_params)
print(params)

# INPUTS
total_iterations = 1400
show_every = 200
resolution = 64
phantom = np.load(f'phantoms/ground_truth/{resolution}/{45}.npy')
phantom_noisy = np.load(f'phantoms/gaussian/resolution_{resolution}/noise_level_.09/pangtom_{45}_gaussian_.09.npy')

model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet', in_channels=1, out_channels=1, init_features=64, pretrained=False)
phantom_noisy = add_gaussian_noise(torch.from_numpy(phantom)[None, :], noise_factor=.09).squeeze(1).numpy()

print(f"\n\n----------------------------------")
print(f'Experiment Configuration:')

print(f'\tTotal Iterations: {total_iterations}')
print(f'\tBurnin Iterations: {params["burnin_iter"]}')
print(f'\tLearning Rate: {params["lr"]}')
print(f'\tImage Resolution: {resolution}')


print(f'\tPlotting every {show_every} iterations')
print(f"----------------------------------\n\n")

# Create the lightning module
module = Eval_SGLD(
                phantom=phantom, 
                phantom_noisy=phantom_noisy,
                lr=params['lr'], 
                burnin_iter=params['burnin_iter'],
                model=model, # model defaults to U-net 
                show_every=show_every
                
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

class test():
    def __init__(self):
        print('test')