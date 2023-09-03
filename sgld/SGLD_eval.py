from __future__ import print_function
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt

import numpy as np
from models import *
import torch
import torch.optim
from torch.optim import Optimizer
from torch.utils.data import Dataset
from torch import tensor
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from utils.common_utils import *

import seaborn as sns

from nni.retiarii.evaluator.pytorch import LightningModule
from nni.retiarii.evaluator.pytorch.lightning import DataLoader

from typing import Any

sns.set_style("darkgrid", {"axes.facecolor": ".9"})

# display images
def np_plot(np_matrix, title, cmap=None):
    plt.clf()
    if cmap is not None:
        fig = plt.imshow(np_matrix.transpose(1, 2, 0), interpolation = 'nearest',cmap='gray')
    else:
        fig = plt.imshow(np_matrix.transpose(1, 2, 0), interpolation = 'nearest')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.title(title)
    plt.axis('off')
    plt.pause(0.05) 


def get_unet():
    return torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet', in_channels=1, out_channels=1, init_features=64, pretrained=False)

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True

# SGLD Pytorch Lightning Module
class SingleImageDataset(Dataset):
    def __init__(self, image, num_iter):
        self.image = image
        self.num_iter = num_iter

    def __len__(self):
        return self.num_iter

    def __getitem__(self, index):
        # Always return the same image (and maybe a noise tensor or other information if necessary??)
        return self.image

class SGLD(LightningModule):
    def __init__(self, 
        original_np,
        noisy_np,
        noisy_torch,
        learning_rate = 0.01,
        show_every=20,
        patience = 1000,
        buffer_size = 100,
        #model=get_unet(),  ### not for NAS
        weight_decay = 5e-8, # this is proportionate to a 1024x1024 image
        model_cls=None,
    ):
        super().__init__()
        self.automatic_optimization = True
        # self.automatic_optimization = False ### not for NAS

        # iterators
        self.burnin_iter=0 # burn-in iteration for SGLD
        self.show_every=show_every
        self.num_iter=20000

        # backtracking
        self.psrn_noisy_last=0
        self.last_net = None
        self.roll_back = True # To solve the oscillation of model training 

        # SGLD Output Accumulation
        self.sgld_mean=0
        self.sgld_mean_each=0
        self.sgld_psnr_list = [] # psnr between sgld out and gt
        self.MCMC_iter=50
        self.param_noise_sigma=2

        # tinker with image input
        self.img_np = original_np           
        self.img_noisy_np = noisy_np
        self.img_noisy_torch = noisy_torch
        
        # network input
        self.input_depth = 1
        
        #self.model = model.type(self.dtype) ### not for NAS
        self.model_cls = model_cls

        self.net_input = get_noise(self.input_depth, 'noise', (self.img_np.shape[-2:][1], self.img_np.shape[-2:][0])).type(self.dtype).detach()
        self.net_input_saved = self.net_input.detach().clone()
        self.noise = self.net_input.detach().clone()
        
        # closure
        self.reg_noise_std = tensor(1./30.)
        self.criteria = torch.nn.MSELoss().type(self.dtype) # loss

        # optimizer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # burnin-end criteria
        self.img_collection = []
        self.variance_history = []
        self.patience = patience
        self.wait_count = 0
        self.best_score = float('inf')
        self.best_epoch = 0
        self.img_collection = []
        self.burnin_over = False
        self.buffer_size = buffer_size
        self.cur_var = None

    def set_model(self, model):
        # This will be called after __init__ and will set the candidate model
        # needed for NAS but not for a standard training loop
        if self.model_cls is not None:
            self.model = self.model_cls
        self.model = model

    def configure_optimizers(self) -> Optimizer:
        """
        We are doing a manual implementation of the SGLD optimizer
        There is a SGLD optimizer that can be found here:
            - https://pysgmcmc.readthedocs.io/en/pytorch/_modules/pysgmcmc/optimizers/sgld.html
            - Implementing this would greatly affect the training step
                - But could it work?? :`( I couldn't figure it out
        """
        return torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

    def train_dataloader(self):
        """
        Trick this puppy into thinking we have a dataloader
        It's a single image for deep image priors
        So we just need to return a dataloader with a single image
        """
        dataset = SingleImageDataset(self.img_np, self.num_iter)
        return DataLoader(dataset, batch_size=1)

    def on_train_start(self) -> None:
        """
        Move all tensors to the GPU to begin training
        Initialize Iterators
        Set Sail
        """
        self.model.to(self.device)
        self.net_input = self.net_input.to(self.device)
        self.img_noisy_torch = self.img_noisy_torch.to(self.device)
        self.reg_noise_std = self.reg_noise_std.to(self.device)

        self.net_input_saved = self.net_input.clone().to(self.device)
        self.noise = self.net_input.clone().to(self.device)
        
        # Initialize Iterations
        self.i=0
        self.sample_count=0

        # bon voyage
        print('Starting optimization with SGLD')

    def forward(self, net_input_saved):
        """
        Forward pass of the model
        occurs in the closure function in this implementation
        """
        if self.reg_noise_std > 0:
            self.net_input = self.net_input_saved + (self.noise.normal_() * self.reg_noise_std)
            return self.model(self.net_input)
        else:
            return self.model(net_input_saved)

    def update_burnin(self,out_np):
        """
        Componenet of closure function
        check if we should end the burnin phase
        """
        # update img collection
        v_img_np = out_np.reshape(-1)
        self.update_img_collection(v_img_np)
        img_collection = self.get_img_collection()

        if len(img_collection) >= self.buffer_size:
            # update variance and var history
            ave_img = np.mean(img_collection, axis=0)
            variance = [self.MSE(ave_img, tmp) for tmp in img_collection]
            self.cur_var = np.mean(variance)
            self.variance_history.append(self.cur_var)
            self.check_stop(self.cur_var, self.i)
    
    def backtracking(self, psrn_noisy, total_loss):
        """
        Componenet of closure function
        backtracking to prevent oscillation if the PSNR is fluctuating
        """
        if self.roll_back and self.i % self.show_every:
            if psrn_noisy - self.psrn_noisy_last < -5: 
                print('Falling back to previous checkpoint.')
                for new_param, net_param in zip(self.last_net, self.model.parameters()):
                    net_param.detach().copy_(new_param.cuda())
                return total_loss*0
            else:
                self.last_net = [x.detach().cpu() for x in self.model.parameters()]
                self.psrn_noisy_last = psrn_noisy

    def closure_sgld(self):
        torch.autograd.set_detect_anomaly(True)
        out = self.forward(self.net_input)

        # compute loss
        total_loss = self.criteria(out, self.img_noisy_torch)
        #total_loss.backward() ### not for NAS
        total_loss.backward(retain_graph=True) # retain_graph=True is for NAS to work
        out_np = out.detach().cpu().numpy()[0]

        # compute PSNR
        psrn_noisy = compare_psnr(self.img_noisy_np, out.detach().cpu().numpy()[0])
        psrn_gt    = compare_psnr(self.img_np, out_np)
        self.sgld_psnr_list.append(psrn_gt)

        # early burn in termination criteria
        if not self.burnin_over:
            self.update_burnin(out_np)

        # # backtracking 
        # self.backtracking(psrn_noisy, total_loss)
        
        # plot progress
        if self.i % self.show_every == 0:
            self.plot_progress(out_np, psrn_gt)

        ##########################################
        ### Logging and SGLD mean collection #####
        ##########################################
        
        if self.burnin_over and np.mod(self.i, self.MCMC_iter) == 0:
            self.sgld_mean += out_np
            self.sample_count += 1.

        if self.burnin_over:
            self.burnin_iter+=1
            self.sgld_mean_each += out_np
            self.sgld_mean_tmp = self.sgld_mean_each / self.burnin_iter # (self.i - self.burnin_iter)
            self.sgld_mean_psnr_each = compare_psnr(self.img_np, self.sgld_mean_tmp)

            if self.i % (self.show_every/5) == 0:
                print('Iter: %d; psnr_gt %.2f; psnr_sgld %.2f' % (self.i, psrn_gt, self.sgld_mean_psnr_each))

        elif self.cur_var is not None and not self.burnin_over:
            if self.i % (self.show_every/5) == 0:
                print('Iter: %d; psnr_gt %.2f; loss %.5f; var %.8f' % (self.i, psrn_gt, total_loss, self.cur_var))

        else:
            if self.i % (self.show_every/5) == 0:
                print('Iter: %d; psnr_gt %.2f; loss %.5f' % (self.i, psrn_gt, total_loss))
        
        # if self.i == self.burnin_iter and self.burnin_over:
        #     print('Burn-in done, start sampling')

        self.i += 1
        return total_loss

    def add_noise(self, net):
        """
        Add noise to the network parameters
        This is the critical part of SGLD
        """
        for n in [x for x in net.parameters() if len(x.size()) == 4]:
            noise = torch.randn(n.size())*self.param_noise_sigma*self.learning_rate
            #noise = noise.type(self.dtype)
            noise = noise.type(self.dtype).to(self.device)
            n.data = n.data + noise

    # Define hook 
    def on_train_batch_start(self, batch, batch_idx):
        optimizer = self.optimizers()[0]
        optimizer.zero_grad()
        # print(f'made it through zero_grad on iter {self.i}')

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """
        Add noise for SGLD
        """
        self.add_noise(self.model)
        # print(f'made it through add_noise on iter {self.i}')


    def training_step(self, batch: Any, batch_idx: int) -> Any:
        """
        Oh the places you'll go
        ---> Straight to error city calling this add_noise in the training step
        ---> Consider using the on_train_batch_end hook? (each batch is only one iteration)
        """
        optimizer = self.optimizers()[0]
        # print attributes about optimizer,
        # i think we're getting a list of one element
        # print(f'opt list count: {len(optimizer)}')
        # print(optimizer)

        # optimizer.zero_grad()
        # print(f'made it through zero_grad on iter {self.i}')

        loss = self.closure_sgld()
        # print(f'made it through closure on iter {self.i}')

        # optimizer.step()
        # print(f'made it through step on iter {self.i}')

        # self.add_noise(self.model)
        # print(f'made it through add_noise on iter {self.i}')

        return loss

    def on_train_end(self) -> None:
        """
        May all your dreams come true
        """
        plotted = self.sgld_mean / self.sample_count
        np_plot(plotted, 'Final after %d iterations' % (self.i), cmap='gray')

    def check_stop(self, current, cur_epoch):
        """
        using an early stopper technique to determine when to end the burn in phase for SGLD
        https://arxiv.org/pdf/2112.06074.pdf
        https://github.com/sun-umn/Early_Stopping_for_DIP/blob/main/ES_WMV.ipynb
        """
        if current < self.best_score:
            self.best_score = current
            self.best_epoch = cur_epoch
            self.wait_count = 0
            self.burnin_over = False
        else:
            self.wait_count += 1
            self.burnin_over = self.wait_count >= self.patience
        if self.burnin_over:
            print(f'\n\nBurn-in completed at iter {self.i}; \nStarting SGLD Mean sampling;\n\n')
            self.show_every = self.MCMC_iter

    def update_img_collection(self, cur_img):
        self.img_collection.append(cur_img)
        if len(self.img_collection) > self.buffer_size:
            self.img_collection.pop(0)

    def get_img_collection(self):
        return self.img_collection

    def MSE(self, x1, x2):
        return ((x1 - x2) ** 2).sum() / x1.size

    def plot_progress(self, out_np, psrn_gt):
        """
        plot original image
        plot denoised image
        plot noisy image

        everything grayscaled
        """
        if self.burnin_over and self.sample_count > 0:
            plotted = self.sgld_mean / self.sample_count
            #plotted = plotted.detach().cpu().numpy()[0]
            label = "SGLD mean"
        else:
            plotted = out_np
            label = "Denoised image"

        _, self.ax = plt.subplots(1, 3, figsize=(10, 5))
        self.ax[0].imshow(self.img_np.transpose(1, 2, 0), interpolation = 'nearest', cmap='gray')
        self.ax[0].set_title('Original image')
        self.ax[0].axis('off')

        self.ax[1].imshow(plotted.transpose(1, 2, 0), interpolation = 'nearest', cmap='gray')
        self.ax[1].set_title(label)
        self.ax[1].axis('off')

        self.ax[2].imshow(self.img_noisy_np.transpose(1, 2, 0), interpolation = 'nearest', cmap='gray')
        self.ax[2].set_title('Noisy image')
        self.ax[2].axis('off')
        
        plt.tight_layout()
        plt.show()

