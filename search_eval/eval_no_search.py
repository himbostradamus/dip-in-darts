from nni import trace, report_intermediate_result, report_final_result
from nni.retiarii.evaluator.pytorch import LightningModule
from nni.retiarii.evaluator.pytorch.lightning import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from skimage.metrics import peak_signal_noise_ratio as compare_psnr

import torch
from torch import tensor

import matplotlib.pyplot as plt

from typing import Any
import numpy as np

from .utils.common_utils import get_noise
from .optimizer.SGLD import SGLD
from optimizer.SingleImageDataset import SingleImageDataset

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
dtype = torch.cuda.FloatTensor

@trace
class LightningEvalSearchSGLD(LightningModule):
    def __init__(self, 
                 phantom=None, 
                 phantom_noisy=None,
                 num_iter=1,
                 lr=0.01,
                 reg_noise_std_val=1./30., 
                 burnin_iter=1800, 
                 MCMC_iter=50,
                 model_cls=None, 
                 show_every=200
                ):
        super().__init__()
        torch.autograd.set_detect_anomaly(True)
        
        # network features
        self.model_cls = model_cls
        self.model = model_cls
        self.input_depth = 1

        # loss
        self.total_loss = 10
        self.criteria = torch.nn.MSELoss().type(dtype)

        # "Early Stopper" Trigger
        self.reg_noise_std = tensor(reg_noise_std_val)
        self.learning_rate = lr
        self.roll_back = True # to prevent numerical issues
        self.num_iter = num_iter # max iterations
        self.burnin_iter = burnin_iter # burn-in iteration for SGLD
        self.weight_decay = 5e-8
        self.show_every =  show_every
        self.report_every = self.show_every / 5

        # SGLD
        self.sgld_mean_each = 0
        self.sgld_psnr_mean_list = []
        self.MCMC_iter = MCMC_iter
        self.sgld_mean = 0
        self.last_net = None
        self.psnr_noisy_last = 0
        self.param_noise_sigma = 2

        # image and noise
        self.phantom = phantom
        self.img_np = phantom
        self.img_noisy_np = phantom_noisy

        self.img_noisy_torch = torch.tensor(self.img_noisy_np, dtype=torch.float32).unsqueeze(0)
        self.net_input = get_noise(self.input_depth, 'noise', (self.img_np.shape[-2:][1], self.img_np.shape[-2:][0])).type(self.dtype).detach()

    def configure_optimizers(self):
        """
        Basic Adam Optimizer
        LR Scheduler: ReduceLROnPlateau
        1 Parameter Group
        """
        print('Starting optimization with SGLD')
        optimizer = SGLD(
                        self.model.parameters(), 
                        lr=self.learning_rate, 
                        num_burn_in_steps=self.burnin_iter, 
                        precondition_decay_rate=1-self.weight_decay
                        )
        return optimizer
    
    def on_train_start(self):
        """
        Move all tensors to the GPU to begin training
        """
        self.model.to(self.device)
        self.net_input = self.net_input.to(self.device)
        self.img_noisy_torch = self.img_noisy_torch.to(self.device)
        self.reg_noise_std = self.reg_noise_std.to(self.device)
        self.sample_count = 0

        self.net_input_saved = self.net_input.clone().to(self.device)
        self.noise = self.net_input.clone().to(self.device)
        self.i = 0
        self.iteration_counter = 0
          
    def forward(self, net_input_saved):
        torch.autograd.set_detect_anomaly(True)
        if self.reg_noise_std > 0:
            net_input = net_input_saved + (self.noise.normal_() * self.reg_noise_std)
            return self.model(net_input)
        else:
            return self.model(net_input_saved)
        
    def closure(self, r_img_torch):
        out = r_img_torch
        self.total_loss = self.criteria(out, self.img_noisy_torch)
        self.latest_loss = self.total_loss.item()
        self.log('loss', self.latest_loss)
        out_np = out.detach().cpu().numpy()[0]

        print(f"\n\nimg_noisy_np type: {self.img_noisy_np.dtype} -- out_np type: {out_np.dtype} \n\n")
        self.psnr_noisy = compare_psnr(self.img_noisy_np, out_np)

        self.log("psrn_noisy", self.psnr_noisy)

        if self.i > self.burnin_iter and np.mod(self.i, self.MCMC_iter) == 0:
            self.sgld_mean += out_np
            self.sample_count += 1.

        if self.i > self.burnin_iter:
            self.sgld_mean_each += out_np
            sgld_mean_tmp = self.sgld_mean_each / (self.i - self.burnin_iter)
            self.sgld_mean_psnr_each = compare_psnr(self.img_np, sgld_mean_tmp)
            self.sgld_psnr_mean_list.append(self.sgld_mean_psnr_each) # record the PSNR of avg after burn-in

        if self.iteration_counter % self.report_every == 0 and self.i > self.burnin_iter:
            report_intermediate_result({
                'iteration': self.iteration_counter ,
                'loss': self.latest_loss, 
                'sample count': self.i - self.burnin_iter, 
                'psnr_sgld_last': self.sgld_psnr_mean_list[-1]})
        elif self.iteration_counter % self.report_every == 0:
            report_intermediate_result({
                'iteration': self.iteration_counter ,
                'loss': round(self.latest_loss,5),
                'psnr_noisy': round(self.psnr_noisy,5)})

        self.i += 1 # this may cause a problem with two iterations per batch

    def training_step(self, batch, batch_idx):
        """
        Deep Image Prior

        training here follows closely from the following two repos: 
            - the deep image prior repo
            - a DIP SGLD repo 
        (NNI needs Lighting Module for the evaluator so this blends the two)
        """      
        self.iteration_counter += 1

        r_img_torch = self.forward(self.net_input)

        self.closure(r_img_torch)

        if self.iteration_counter % self.show_every == 0:
            self.plot_progress()
        return {"loss": self.total_loss}

    def on_train_batch_end(self, outputs, batch, batch_idx, *args, **kwargs):
        """
        Add noise for SGLD
        """
        # Get current optimizer
        optimizer = self.optimizers()

        # if the optimizer is Adam, add noise
        if isinstance(optimizer, torch.optim.Adam):
            self.add_noise(self.model)
        # if the opinion is SGLD, do nothing

    def on_train_end(self, **kwargs: Any):
        """
        Report final metrics and display the results
        """
        # final log
        report_final_result({'loss': self.latest_loss})

        # # plot images to see results
        self.plot_progress()
        final_sgld_mean = self.sgld_mean / self.sample_count
        final_sgld_mean_psnr = compare_psnr(self.img_np, final_sgld_mean)
        print(f"Final SGLD mean PSNR: {final_sgld_mean_psnr}")

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
    
    def add_noise(self, net):
        for n in [x for x in net.parameters() if len(x.size()) == 4]:
            noise = torch.randn(n.size())*self.param_noise_sigma*self.learning_rate
            noise = noise.to(self.device)
            n.data = n.data + noise
        
    def plot_progress(self):
        if self.i < self.burnin_iter*1.1:
            denoised_img = self.forward(self.net_input).detach().cpu().squeeze().numpy()
        else:
            print(f'MCMC sample count: {self.sample_count}')
            denoised_img = self.sgld_mean_each / (self.i - self.burnin_iter)
            # denoised_img = self.sgld_mean / self.sample_count if self.sample_count > 0 else self.sgld_mean
            denoised_img = np.squeeze(denoised_img)

        _, self.ax = plt.subplots(1, 3, figsize=(10, 5))
        
        #self.ax[0].clear()
        #self.ax[1].clear()
        #self.ax[2].clear()

        self.ax[0].imshow(self.img_np.squeeze(), cmap='gray')
        self.ax[0].set_title("Original Image")
        self.ax[0].axis('off')

        self.ax[1].imshow(denoised_img, cmap='gray')
        self.ax[1].set_title("Denoised Image")
        self.ax[1].axis('off')

        self.ax[2].imshow(self.img_noisy_torch.detach().cpu().squeeze().numpy(), cmap='gray')
        self.ax[2].set_title("Noisy Image")
        self.ax[2].axis('off')

        #clear_output(wait=True)

        plt.tight_layout()
        plt.show()
