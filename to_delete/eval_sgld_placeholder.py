import numpy as np
import torch
from skimage.measure import compare_psnr

from torch.utils.data import Dataset
from torch import optim, tensor, zeros_like

import pytorch_lightning as pl

from nni import trace, report_intermediate_result, report_final_result
import nni.retiarii.nn.pytorch as nn
from nni.retiarii.evaluator.pytorch import LightningModule
from nni.retiarii.evaluator.pytorch.lightning import DataLoader
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from typing import Any

from darts.common_utils import *
from darts.noises import add_selected_noise
from darts.phantom import generate_phantom

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
dtype = torch.cuda.FloatTensor


reg_noise_std = 1./30.
learning_rate = LR = 0.01
exp_weight=0.99
input_depth = 32 
roll_back = True # to prevent numerical issues
num_iter = 20000 # max iterations
burnin_iter = 7000 # burn-in iteration for SGLD
weight_decay = 5e-8
show_every =  500
mse = torch.nn.MSELoss().type(dtype) # loss


sgld_psnr_list = [] # psnr between sgld out and gt
sgld_mean = 0
roll_back = True # To solve the oscillation of model training 
last_net = None
psrn_noisy_last = 0
MCMC_iter = 50
param_noise_sigma = 2

sgld_mean_each = 0
sgld_psnr_mean_list = [] # record the PSNR of avg after burn-in

###
###
###

# adjusting input
def preprocess_image(resolution, noise_type, noise_factor, input_img_np=None):
    """
    Generates an image (or takes an input phantom), adds noise, and converts it to both numpy and torch tensors.

    Args:
    - resolution (int): Resolution for the phantom image.
    - noise_type (str): Type of noise to add.
    - noise_factor (float): Noise factor.
    - input_img_np (numpy.ndarray, optional): Input raw image in numpy format. If not provided, a new image will be generated.

    Returns:
    - img_np (numpy.ndarray): Original image in numpy format.
    - img_noisy_np (numpy.ndarray): Noisy image in numpy format.
    - img_torch (torch.Tensor): Original image in torch tensor format.
    - img_noisy_torch (torch.Tensor): Noisy image in torch tensor format.
    """
    if input_img_np is None:
        raw_img_np = generate_phantom(resolution=resolution) # 1x64x64 np array
    else:
        raw_img_np = input_img_np.copy() # 1x64x64 np array
        
    img_np = raw_img_np.copy() # 1x64x64 np array
    img_torch = torch.tensor(raw_img_np, dtype=torch.float32).unsqueeze(0) # 1x1x64x64 torch tensor
    img_noisy_torch = add_selected_noise(img_torch, noise_type=noise_type, noise_factor=noise_factor) # 1x1x64x64 torch tensor
    img_noisy_np = img_noisy_torch.squeeze(0).numpy() # 1x64x64 np array
    
    return img_np, img_noisy_np, img_torch, img_noisy_torch

noise_type='gaussian'
noise_factor=0.05 
resolution = 6
max_depth = resolution - 1
phantom = generate_phantom(resolution=resolution)
if phantom is None:
    img_np, img_noisy_np, _, img_noisy_torch = preprocess_image(resolution, noise_type, noise_factor)
else:
    img_np, _, _, img_noisy_torch = preprocess_image(resolution, noise_type, noise_factor, input_img_np=phantom)
net_input = get_noise(input_depth=1, spatial_size=img_np.shape[1], noise_type=noise_type).type(dtype).detach()

###
###
###


## SGLD
def add_noise(model):
    for n in [x for x in model.parameters() if len(x.size()) == 4]:
        noise = torch.randn(n.size())*param_noise_sigma*learning_rate
        noise = noise.type(dtype)
        n.data = n.data + noise

net = _net

## Input random noise
net_input_saved = net_input.detach().clone()
noise = net_input.detach().clone()
i = 0

sample_count = 0

def closure_sgld():
    global i, net_input, sgld_mean, sample_count, psrn_noisy_last, last_net, sgld_mean_each
    if reg_noise_std > 0:
        net_input = net_input_saved + (noise.normal_() * reg_noise_std)
    out = net(net_input)
    total_loss = mse(out, img_noisy_torch)
    total_loss.backward()
    out_np = out.detach().cpu().numpy()[0]

    psrn_noisy = compare_psnr(img_noisy_np, out.detach().cpu().numpy()[0])

    # Backtracking
    if roll_back and i % show_every:
        if psrn_noisy - psrn_noisy_last < -5: 
            print('Falling back to previous checkpoint.')
            for new_param, net_param in zip(last_net, net.parameters()):
                net_param.detach().copy_(new_param.cuda())
            return total_loss*0
        else:
            last_net = [x.detach().cpu() for x in net.parameters()]
            psrn_noisy_last = psrn_noisy
    
    if i > burnin_iter and np.mod(i, MCMC_iter) == 0:
        sgld_mean += out_np
        sample_count += 1.

    i += 1
    return total_loss


  ## Optimizing 
print('Starting optimization with SGLD')
optimizer = optim.Adam(net.parameters(), lr=LR, weight_decay = weight_decay)
for j in range(num_iter):
    optimizer.zero_grad()
    closure_sgld()
    optimizer.step()
    add_noise(net)