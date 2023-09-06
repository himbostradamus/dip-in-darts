# this is the HPO Search Space

import nni
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

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

from phantom import generate_phantom

from pytorch_lightning.callbacks import ModelCheckpoint

from nni.nas.evaluator.pytorch import Lightning, Trainer, LightningModule
from nni.nas.evaluator.pytorch.lightning import DataLoader

from typing import Any


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True

def get_unet():
    return torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet', in_channels=1, out_channels=1, init_features=64, pretrained=False)

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

class SGLD_HPO(LightningModule):
    def __init__(self, 
        original_np,
        noisy_np,
        noisy_torch,
        learning_rate = 0.01,
        show_every=20,
        patience = 1000,
        buffer_size = 100,
        model=get_unet(),
        weight_decay=5e-8,

    ):
        super().__init__()
        self.automatic_optimization = False

        # iterators
        self.burnin_iter=0 # burn-in iteration for SGLD
        self.show_every=show_every
        self.num_iter=100 # max iterations

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
        self.model = model.type(self.dtype)
        self.net_input = get_noise(self.input_depth, 'noise', (img_np.shape[-2:][1], img_np.shape[-2:][0])).type(self.dtype).detach()
        self.net_input_saved = self.net_input.detach().clone()
        self.noise = self.net_input.detach().clone()
        
        # closure
        self.reg_noise_std = tensor(1./30.)
        self.criteria = torch.nn.MSELoss().type(dtype) # loss

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
        out = self.forward(self.net_input)

        # compute loss
        total_loss = self.criteria(out, self.img_noisy_torch)
        total_loss.backward()
        out_np = out.detach().cpu().numpy()[0]

        # compute PSNR
        psrn_noisy = compare_psnr(self.img_noisy_np, out.detach().cpu().numpy()[0])
        psrn_gt    = compare_psnr(self.img_np, out_np)
        self.sgld_psnr_list.append(psrn_gt)

        # early burn in termination criteria
        if not self.burnin_over:
            self.update_burnin(out_np)

        # backtracking 
        self.backtracking(psrn_noisy, total_loss)

        ##########################################
        ### Logging and SGLD mean collection #####
        ##########################################
        
        if self.burnin_over and np.mod(self.i, self.MCMC_iter) == 0:
            self.sgld_mean += out_np
            self.sample_count += 1.
            sgld_psnr = compare_psnr(self.img_np, self.sgld_mean / self.sample_count)
            # self.log({'loss': total_loss, 'psnr_gt': psrn_gt, 'psnr_sgld': sgld_psnr})
            # nni.report_intermediate_result({'loss': total_loss, 'psnr_gt': psrn_gt, 'psnr_sgld': sgld_psnr})

        if self.burnin_over:
            self.burnin_iter+=1
            self.sgld_mean_each += out_np

        # elif self.cur_var is not None and not self.burnin_over:
        #     self.log({'loss': total_loss, 'psnr_gt': psrn_gt, 'var': self.cur_var})
        #     nni.report_intermediate_result({'loss': total_loss, 'psnr_gt': psrn_gt, 'var': self.cur_var})

        # else:
        #     self.log({'loss': total_loss, 'psnr_gt': psrn_gt})
        #     nni.report_intermediate_result({'loss': total_loss, 'psnr_gt': psrn_gt})

        self.i += 1
        self.log('psnr',psrn_gt)
        nni.report_final_result('psnr',psrn_gt)
        return total_loss

    def add_noise(self, net):
        """
        Add noise to the network parameters
        This is the critical part of SGLD
        """
        for n in [x for x in net.parameters() if len(x.size()) == 4]:
            noise = torch.randn(n.size())*self.param_noise_sigma*self.learning_rate
            noise = noise.type(dtype)
            n.data = n.data + noise

    def training_step(self, batch: Any, batch_idx: int) -> Any:
        """
        Oh the places you'll go
        ---> Straight to error city calling this add_noise in the training step
        ---> Consider using the on_train_batch_end hook? (each batch is only one iteration)
        """
        optimizer = self.optimizers()
        optimizer.zero_grad()
        loss = self.closure_sgld()
        optimizer.step()
        self.add_noise(self.model)
        return loss

    def on_train_end(self) -> None:
        """
        May all your dreams come true
        """
        # get output by sending net_input_saved through the network
        # compute PSNR
        out = self.forward(self.net_input_saved)
        out_np = out.detach().cpu().numpy()[0]
        psrn_gt    = compare_psnr(self.img_np, out_np)
        
        # compute SGLD mean from MCMC samples
        sgld_final = self.sgld_mean / self.sample_count
        sgld_final_psnr = compare_psnr(self.img_np, sgld_final)
        
        # compute SGLD mean from all post burnin samples
        self.sgld_mean_tmp = self.sgld_mean_each / self.burnin_iter
        sgld_final_psnr_tmp = compare_psnr(self.img_np, self.sgld_mean_tmp)
        
        # self.log({'psnr_gt': psrn_gt, 'psnr_sgld': sgld_final_psnr, 'psnr_sgld_each': sgld_final_psnr_tmp})
        # nni.report_final_result({'psnr_gt': psrn_gt, 'psnr_sgld': sgld_final_psnr, 'psnr_sgld_each': sgld_final_psnr_tmp})
        self.log('psnr',psrn_gt)
        nni.report_final_result('psnr',psrn_gt)

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

params = {
        'learning_rate': 0.01,
        'patience': 1000,
        'buffer_size': 100,
        'weight_decay': 5e-8, # this is proportionate to a 1024x1024 image
        }

optimized_params = nni.get_next_parameter()
params.update(optimized_params)
print(params)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# check if CUDA is available
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor 

# choose iterations
num_iter = 1e4 # max iterations

# get image
# generate phantom stored in subfolder of parent directory
resolution = 6
max_depth = resolution - 1
phantom = generate_phantom(resolution=resolution)
raw_img_np = phantom.copy() # 1x64x64 np array    
img_np = raw_img_np.copy() # 1x64x64 np array
sigma=25/255
# sigma = .05
img_noisy_np = np.clip(img_np + np.random.normal(scale=sigma, size=img_np.shape), 0, 1).astype(np.float32)
img_noisy_torch = np_to_torch(img_noisy_np).type(dtype)

# reference model 
model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
                       in_channels=1, out_channels=1, init_features=64, pretrained=False)

# Create the lightning module
module = SGLD_HPO(
        original_np=img_np,
        noisy_np=img_noisy_np,
        noisy_torch=img_noisy_torch,
        learning_rate = params['learning_rate'],
        patience=params['patience'],
        buffer_size=params['buffer_size'],
        weight_decay=params['weight_decay'],
        )

# Create a PyTorch Lightning trainer
trainer = Trainer(
            max_epochs=num_iter,
            fast_dev_run=False,
            gpus=1,
            checkpoint_callback=False
            )

# Initialize ModelCheckpoint callback
checkpoint_callback = ModelCheckpoint(
    dirpath='./{lightning_logs}/{logger_name}/version_{version}/checkpoints/',
    filename='{epoch}-{step}',
    every_n_epochs=100,
    save_top_k=1,
)

# Add the checkpoint callback to trainer
trainer.callbacks.append(checkpoint_callback)
            
if not hasattr(trainer, 'optimizer_frequencies'):
    trainer.optimizer_frequencies = []

# Create the lighting object for evaluator
train_loader = DataLoader(SingleImageDataset(img_noisy_np, num_iter=1), batch_size=1)
val_loader = DataLoader(SingleImageDataset(img_noisy_np, num_iter=1), batch_size=1)

lightning = Lightning(lightning_module=module, trainer=trainer, train_dataloaders=train_loader, val_dataloaders=val_loader)
lightning.fit(model)