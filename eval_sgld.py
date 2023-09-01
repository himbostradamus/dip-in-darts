from __future__ import print_function
import warnings
warnings.filterwarnings("ignore")
import matplotlib
import matplotlib.pyplot as plt
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import sys
import numpy as np
from models import *
import torch
import torch.optim
import time
from skimage.measure import compare_psnr
from utils.denoising_utils import *
import _pickle as cPickle
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
dtype = torch.cuda.FloatTensor
import seaborn as sns
sns.set_style("darkgrid", {"axes.facecolor": ".9"})

# display images
def np_plot(np_matrix, title):
    plt.clf()
    fig = plt.imshow(np_matrix.transpose(1, 2, 0), interpolation = 'nearest')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.title(title)
    plt.axis('off')
    plt.pause(0.05) 

INPUT = 'noise'
pad = 'reflection'
OPT_OVER = 'net' # optimize over the net parameters only
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
img_noisy_torch = np_to_torch(img_noisy_np).type(dtype)

sgld_psnr_list = [] # psnr between sgld out and gt
sgld_mean = 0
roll_back = True # To solve the oscillation of model training 
last_net = None
psrn_noisy_last = 0
MCMC_iter = 50
param_noise_sigma = 2

sgld_mean_each = 0
sgld_psnr_mean_list = [] # record the PSNR of avg after burn-in

## SGLD
def add_noise(model):
    for n in [x for x in model.parameters() if len(x.size()) == 4]:
        noise = torch.randn(n.size())*param_noise_sigma*learning_rate
        noise = noise.type(dtype)
        n.data = n.data + noise

net2 = get_net(input_depth, 'skip', pad,
            skip_n33d=128, 
            skip_n33u=128,
            skip_n11=4,
            num_scales=5,
            upsample_mode='bilinear').type(dtype)

## Input random noise
net_input = get_noise(input_depth, INPUT, (img_pil.size[1], img_pil.size[0])).type(dtype).detach()
net_input_saved = net_input.detach().clone()
noise = net_input.detach().clone()
i = 0

sample_count = 0

def closure_sgld():
    global i, net_input, sgld_mean, sample_count, psrn_noisy_last, last_net, sgld_mean_each
    if reg_noise_std > 0:
        net_input = net_input_saved + (noise.normal_() * reg_noise_std)
    out = net2(net_input)
    total_loss = mse(out, img_noisy_torch)
    total_loss.backward()
    out_np = out.detach().cpu().numpy()[0]

    psrn_noisy = compare_psnr(img_noisy_np, out.detach().cpu().numpy()[0])
    psrn_gt    = compare_psnr(img_np, out_np)

    sgld_psnr_list.append(psrn_gt)

    # Backtracking
    if roll_back and i % show_every:
        if psrn_noisy - psrn_noisy_last < -5: 
            print('Falling back to previous checkpoint.')
            for new_param, net_param in zip(last_net, net2.parameters()):
                net_param.detach().copy_(new_param.cuda())
            return total_loss*0
        else:
            last_net = [x.detach().cpu() for x in net2.parameters()]
            psrn_noisy_last = psrn_noisy

    if i % show_every == 0:
        np_plot(out.detach().cpu().numpy()[0], 'Iter: %d; gt %.2f' % (i, psrn_gt))
    
    if i > burnin_iter and np.mod(i, MCMC_iter) == 0:
        sgld_mean += out_np
        sample_count += 1.

    if i > burnin_iter:
        sgld_mean_each += out_np
        sgld_mean_tmp = sgld_mean_each / (i - burnin_iter)
        sgld_mean_psnr_each = compare_psnr(img_np, sgld_mean_tmp)
        sgld_psnr_mean_list.append(sgld_mean_psnr_each) # record the PSNR of avg after burn-in

    i += 1
    return total_loss


  ## Optimizing 
print('Starting optimization with SGLD')
optimizer = torch.optim.Adam(net2.parameters(), lr=LR, weight_decay = weight_decay)
for j in range(num_iter):
    optimizer.zero_grad()
    closure_sgld()
    optimizer.step()
    add_noise(net2)

sgld_mean = sgld_mean / sample_count
sgld_mean_psnr = compare_psnr(img_np, sgld_mean)