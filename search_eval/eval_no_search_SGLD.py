from nni import trace, report_intermediate_result, report_final_result
from nni.retiarii.evaluator.pytorch import LightningModule
from nni.retiarii.evaluator.pytorch.lightning import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from skimage.metrics import peak_signal_noise_ratio as compare_psnr

import torch
from torch.optim import Optimizer
from torch import tensor

import matplotlib.pyplot as plt

from typing import Any
import numpy as np

from .utils.common_utils import get_noise
from .optimizer.SingleImageDataset import SingleImageDataset

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
dtype = torch.cuda.FloatTensor

@trace
class Eval_SGLD(LightningModule):
    def __init__(self, 
                 phantom=None, 
                 phantom_noisy=None,

                 lr=0.01,
                 burnin_iter=1800, 
                 weight_decay=5e-8,

                 MCMC_iter=50,
                 reg_noise_std_val=1./30., 
                 show_every=200,
                 model=None, 
                 HPO=False
                ):
        super().__init__()
        self.automatic_optimization = True
        self.HPO = HPO
        
        # network features
        if model is None:
            model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet', in_channels=1, out_channels=1, init_features=64, pretrained=False)
        self.model_cls = model
        self.model = model
        self.input_depth = 1

        # loss
        self.total_loss = 10
        self.criteria = torch.nn.MSELoss().type(dtype)

        # "Early Stopper" Trigger
        self.reg_noise_std = tensor(reg_noise_std_val)
        self.learning_rate = lr
        self.roll_back = True # to prevent numerical issues
        self.burnin_iter = burnin_iter # burn-in iteration for SGLD
        self.weight_decay = weight_decay
        self.show_every =  show_every

        # SGLD
        self.sgld_mean_each = 0
        self.sgld_psnr_mean_list = []
        self.MCMC_iter = MCMC_iter
        self.sgld_mean = 0
        self.last_net = None
        self.psnr_noisy_last = 0
        self.param_noise_sigma = 2

        # image and noise
        # move to float 32 instead of float 64

        self.phantom = np.float32(phantom)
        self.img_np = np.float32(phantom)
        self.img_noisy_np = np.float32(phantom_noisy)

        self.img_noisy_torch = torch.tensor(self.img_noisy_np, dtype=torch.float32).unsqueeze(0)
        self.net_input = get_noise(self.input_depth, 'noise', (self.img_np.shape[-2:][1], self.img_np.shape[-2:][0])).type(self.dtype).detach()

    def configure_optimizers(self) -> Optimizer:
        """
        We are doing a manual implementation of the SGLD optimizer
        There is a SGLD optimizer that can be found here:
            - https://pysgmcmc.readthedocs.io/en/pytorch/_modules/pysgmcmc/optimizers/sgld.html
            - Implementing this would greatly simplify the training step
                - But could it work?? :`( I couldn't figure it out
        """
        return torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
    
    def on_train_start(self):
        """
        Move all tensors to the GPU to begin training
        """
        self.model.to(self.device)
        self.net_input = self.net_input.to(self.device)
        self.img_noisy_torch = self.img_noisy_torch.to(self.device)
        self.reg_noise_std = self.reg_noise_std.to(self.device)

        self.net_input_saved = self.net_input.clone().to(self.device)
        self.noise = self.net_input.clone().to(self.device)
        self.i = 0
        self.sample_count = 0
        # self.iteration_counter = 0
        self.plot_progress()
          
    def forward(self, net_input_saved):
        if self.reg_noise_std > 0:
            net_input = net_input_saved + (self.noise.normal_() * self.reg_noise_std)
            return self.model(net_input)
        else:
            return self.model(net_input_saved)
        
    def closure(self):
        out = self.forward(self.net_input)

        self.total_loss = self.criteria(out, self.img_noisy_torch)
        # self.total_loss.backward()
        self.latest_loss = self.total_loss.item()
        self.log('loss', self.latest_loss)
        out_np = out.detach().cpu().numpy()[0]

        self.psnr_noisy = compare_psnr(self.img_noisy_np, out_np)
        self.psnr_gt    = compare_psnr(self.img_np, out_np)
        self.log("psrn_noisy", self.psnr_noisy)

        if self.i > self.burnin_iter and np.mod(self.i, self.MCMC_iter) == 0:
            self.sgld_mean += out_np
            self.sample_count += 1.
            self.sgld_mean_psnr = compare_psnr(self.img_np, self.sgld_mean / self.sample_count)

        if self.i > self.burnin_iter:
            self.sgld_mean_each += out_np
            sgld_mean_tmp = self.sgld_mean_each / (self.i - self.burnin_iter)
            self.sgld_mean_psnr_each = compare_psnr(self.img_np, sgld_mean_tmp)
            self.sgld_psnr_mean_list.append(self.sgld_mean_psnr_each) # record the PSNR of avg after burn-in

        if not self.HPO:
            if self.i % 10 == 0 and self.i > self.burnin_iter:
                report_intermediate_result({
                    'iteration': self.i ,
                    'loss': round(self.latest_loss,5), 
                    'sample count': self.i - self.burnin_iter, 
                    'psnr_sgld_last': round(self.sgld_psnr_mean_list[-1],5),
                    'psnr_gt': round(self.psnr_gt,5)
                    })
            elif self.i % 10 == 0:
                report_intermediate_result({
                    'iteration': self.i ,
                    'loss': round(self.latest_loss,5),
                    'psnr_noisy': round(self.psnr_noisy,5),
                    'psnr_gt': round(self.psnr_gt,5)
                    })

        self.i += 1 # this may cause a problem with two iterations per batch

        return self.total_loss

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

    def training_step(self, batch: Any, batch_idx: int) -> Any:
        """
        Oh the places you'll go
        ---> Straight to error city calling this add_noise in the training step
        ---> Consider using the on_train_batch_end hook? (each batch is only one iteration)
        """
        # optimizer = self.optimizers()
        # optimizer.zero_grad()
        loss = self.closure()
        # optimizer.step()

        if self.HPO and self.i < self.burnin_iter and self.i % self.show_every == 0:
            report_intermediate_result(round(self.psnr_gt,5))
        if self.HPO and self.i > self.burnin_iter and self.i % self.show_every == 0 and self.sample_count > 0:
            report_intermediate_result(round(self.sgld_mean_psnr,5))

        return {"loss": loss}

    def on_train_batch_end(self, outputs, batch, batch_idx, *args, **kwargs):
        """
        Add noise for SGLD
        """
        optimizer = self.optimizers()
        if isinstance(optimizer, torch.optim.Adam):
            self.add_noise(self.model)

        if self.i % self.show_every == 0:
            self.plot_progress()

    def on_train_end(self, **kwargs: Any):
        """
        Report final metrics and display the results
        """
        if not self.HPO:
            report_final_result({'loss': self.latest_loss})        
            # # plot images to see results
            self.plot_progress()
            final_sgld_mean = self.sgld_mean / self.sample_count
            final_sgld_mean_psnr = compare_psnr(self.img_np, final_sgld_mean)
            print(f"Final SGLD mean PSNR: {final_sgld_mean_psnr}")
        if self.HPO and self.sample_count > 0:
            report_final_result(round(self.sgld_mean_psnr,5))
        if self.HPO and self.sample_count == 0:
            report_final_result(round(self.psnr_gt,5))

    def common_dataloader(self):
        # dataset = SingleImageDataset(self.phantom, self.num_iter)
        dataset = SingleImageDataset(self.phantom, 1)
        return DataLoader(dataset, batch_size=1)

    def train_dataloader(self):
        return self.common_dataloader()
    
    def optimizer_zero_grad(self, epoch, batch_idx, optimizer, opt_idx):
        # Not sure if this is the default logic in the nni.retiarii.evaluator.pytorch.LightningModule
        # needed to modify so it can accept the opt_idx argument
        optimizer.zero_grad()
    
    def configure_gradient_clipping(self, optimizer, opt_idx, gradient_clip_val, gradient_clip_algorithm):
        # Not sure if this is the default logic in the nni.retiarii.evaluator.pytorch.LightningModule
        # needed to modify so it can accept the opt_idx argument
        # now need to define the clipping logic
        self.clip_gradients(
            optimizer,
            gradient_clip_val=gradient_clip_val,
            gradient_clip_algorithm=gradient_clip_algorithm
        )
        
    def plot_progress(self):
        if self.i < self.burnin_iter+1:
            denoised_img = self.forward(self.net_input).detach().cpu().squeeze().numpy()
            label = 'Denoised Image'
        else:
            print(f'MCMC sample count: {self.sample_count}')
            denoised_img = self.sgld_mean_each / (self.i - self.burnin_iter)
            # denoised_img = self.sgld_mean / self.sample_count if self.sample_count > 0 else self.sgld_mean
            denoised_img = np.squeeze(denoised_img)
            label = 'SGLD Mean'

        _, self.ax = plt.subplots(1, 3, figsize=(10, 5))
        
        #self.ax[0].clear()
        #self.ax[1].clear()
        #self.ax[2].clear()

        self.ax[0].imshow(self.img_np.squeeze(), cmap='gray')
        self.ax[0].set_title("Original Image")
        self.ax[0].axis('off')

        self.ax[1].imshow(denoised_img, cmap='gray')
        self.ax[1].set_title(label)
        self.ax[1].axis('off')

        self.ax[2].imshow(self.img_noisy_torch.detach().cpu().squeeze().numpy(), cmap='gray')
        self.ax[2].set_title("Noisy Image")
        self.ax[2].axis('off')

        #clear_output(wait=True)

        plt.tight_layout()
        plt.show()
