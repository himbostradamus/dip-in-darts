import nni
from nni.retiarii.evaluator.pytorch import Lightning, Trainer
from nni.retiarii.evaluator.pytorch.lightning import DataLoader

import numpy as np
import torch

import sys
sys.path.insert(1, '/home/joe/nas-for-dip')
from search_eval.utils.common_utils import *
from search_eval.eval_no_search_SGLD import Eval_SGLD, SingleImageDataset
from search_space.old.attention_space import DeepImagePrior

# make sure cuda is available and ready to go
torch.cuda.empty_cache()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
print('CUDA available: {}'.format(torch.cuda.is_available()))

params = {
    'max_iter': 1000,
    'lr': 0.1,
    'burnin_iter': 700,

}

optimized_params = nni.get_next_parameter()
params.update(optimized_params)
print(params)

# INPUTS
show_every = 2000
report_every = 250
resolution = 64
noise_type = 'gaussian'
noise_level = '0.09'
phantom =       np.load(f'/home/joe/nas-for-dip/phantoms/ground_truth/{resolution}/{45}.npy')
phantom_noisy = np.load(f'/home/joe/nas-for-dip/phantoms/{noise_type}/res_{resolution}/nl_{noise_level}/p_{45}.npy')

model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet', in_channels=1, out_channels=1, init_features=64, pretrained=False)
model = DeepImagePrior(1,1,5)

print(f"\n\n----------------------------------")
print(f'Experiment Configuration:')

print(f'\tTotal Iterations: {params["max_iter"]}')
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
                show_every=show_every,
                report_every=report_every,
                HPO=True

                )

# Create a PyTorch Lightning trainer
trainer = Trainer(
            max_epochs=params['max_iter'],
            fast_dev_run=False,
            gpus=1,
            )
            
if not hasattr(trainer, 'optimizer_frequencies'):
    trainer.optimizer_frequencies = []

# Create the lighting object for evaluator
train_loader = DataLoader(SingleImageDataset(phantom, num_iter=1), batch_size=1)

lightning = Lightning(lightning_module=module, trainer=trainer, train_dataloaders=train_loader, val_dataloaders=None)
lightning.fit(model)