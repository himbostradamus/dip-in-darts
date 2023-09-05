import matplotlib.pyplot as plt
import numpy as np

from nni import trace, report_intermediate_result, report_final_result
from nni.retiarii.evaluator.pytorch import LightningModule
from nni.retiarii.evaluator.pytorch.lightning import DataLoader

from skimage.metrics import peak_signal_noise_ratio as compare_psnr

import torch
from torch.optim import Optimizer
import torch.optim
from torch import tensor
from typing import Any

from .utils.common_utils import get_noise
from .optimizer.SingleImageDataset import SingleImageDataset

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
dtype = torch.cuda.FloatTensor

@trace
class Eval_OS(LightningModule):
    def __init__(self, 
                 phantom,
                 phantom_noisy,
                 
                 learning_rate = 0.01,
                 patience = 1000,
                 buffer_size = 100,
                 weight_decay = 5e-8, # this is proportionate to a 1024x1024 image

                 MCMC_iter=50, 
                 show_every=20,
                 model_cls=None,
                 HPO=False
            ):
        super().__init__()
        self.automatic_optimization = True
        self.HPO = HPO

        # iterators
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
        self.MCMC_iter=MCMC_iter
        self.param_noise_sigma=2

        # tinker with image input
        self.img_np = phantom           
        self.img_noisy_np = phantom_noisy
        self.img_noisy_torch = torch.tensor(self.img_noisy_np, dtype=torch.float32).unsqueeze(0)
        
        # network input
        self.input_depth = 1
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

    def configure_optimizers(self) -> Optimizer:
        """
        We are doing a manual implementation of the SGLD optimizer
        There is a SGLD optimizer that can be found here:
            - https://pysgmcmc.readthedocs.io/en/pytorch/_modules/pysgmcmc/optimizers/sgld.html
            - Implementing this would greatly affect the training step
                - But could it work?? :`( I couldn't figure it out
        """
        return torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

    def set_model(self, model):
        if self.model_cls is not None:
            self.model = self.model_cls
        self.model = model

    def on_train_start(self) -> None:
        """
        Move all tensors to the GPU to begin training
        Initialize Iterators
        Set Sail
        """
        # move to device
        self.model.to(self.device)
        self.net_input = self.net_input.to(self.device)
        self.img_noisy_torch = self.img_noisy_torch.to(self.device)
        self.reg_noise_std = self.reg_noise_std.to(self.device)

        self.net_input_saved = self.net_input.clone().to(self.device)
        self.noise = self.net_input.clone().to(self.device)
        
        # Initialize Iterations
        self.i=0
        self.sample_count=0
        self.burnin_iter=0 # burn-in iteration for SGLD
        self.report_every = self.show_every/5

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

    def closure(self):
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
            self.sgld_mean_tmp = self.sgld_mean_each / self.burnin_iter
            self.sgld_mean_psnr_each = compare_psnr(self.img_np, self.sgld_mean_tmp)

            if self.i % self.report_every == 0:
                print('Iter: %d; psnr_gt %.2f; psnr_sgld %.2f' % (self.i, psrn_gt, self.sgld_mean_psnr_each))

        elif self.cur_var is not None and not self.burnin_over:
            if self.i % self.report_every == 0:
                print('Iter: %d; psnr_gt %.2f; loss %.5f; var %.8f' % (self.i, psrn_gt, total_loss, self.cur_var))

        else:
            if self.i % self.report_every == 0:
                print('Iter: %d; psnr_gt %.2f; loss %.5f' % (self.i, psrn_gt, total_loss))
        
        self.i += 1
        return total_loss

    def add_noise(self, net):
        """
        Add noise to the network parameters
        This is the critical part of SGLD
        """
        for n in [x for x in net.parameters() if len(x.size()) == 4]:
            noise = torch.randn(n.size())*self.param_noise_sigma*self.learning_rate
            noise = noise.type(self.dtype).to(self.device)
            n.data = n.data + noise

    # Define hook 
    def on_train_batch_start(self, batch, batch_idx):
        optimizer = self.optimizers()[0]
        optimizer.zero_grad()

    def training_step(self, batch: Any, batch_idx: int) -> Any:
        """
        Oh the places you'll go
        """
        loss = self.closure()
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """
        Add noise for SGLD
        """
        self.add_noise(self.model)

    def on_train_end(self) -> None:
        """
        May all your dreams come true
        """
        if not self.HPO:
            self.plot_progress()
            final_sgld_mean = self.sgld_mean / self.sample_count
            final_sgld_mean_psnr = compare_psnr(self.img_np, final_sgld_mean)
            print(f"Final SGLD mean PSNR: {round(final_sgld_mean_psnr,5)}")
            report_final_result(final_sgld_mean_psnr)
        if self.HPO and self.sample_count != 0:
            report_final_result(round(self.sgld_mean_psnr,5))
        if self.HPO and self.sample_count == 0:
            report_final_result(round(self.psnr_gt,5))

    def common_dataloader(self):
        # dataset = SingleImageDataset(self.phantom, self.num_iter)
        dataset = SingleImageDataset(self.phantom, 1)
        return DataLoader(dataset, batch_size=1)

    def train_dataloader(self):
        return self.common_dataloader()

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
        if self.sample_count == 0:
            denoised_img = self.forward(self.net_input).detach().cpu().squeeze().numpy()
            label = "Denoised Image"
        else:
            denoised_img = self.sgld_mean_each / self.burnin_iter
            # denoised_img = self.sgld_mean / self.sample_count if self.sample_count > 0 else self.sgld_mean
            denoised_img = np.squeeze(denoised_img)
            label = "SGLD Mean"

        _, self.ax = plt.subplots(1, 3, figsize=(10, 5))
        self.ax[0].imshow(self.img_np.transpose(1, 2, 0), interpolation = 'nearest', cmap='gray')
        self.ax[0].set_title('Original image')
        self.ax[0].axis('off')

        self.ax[1].imshow(denoised_img.transpose(1, 2, 0), interpolation = 'nearest', cmap='gray')
        self.ax[1].set_title(label)
        self.ax[1].axis('off')

        self.ax[2].imshow(self.img_noisy_np.transpose(1, 2, 0), interpolation = 'nearest', cmap='gray')
        self.ax[2].set_title('Noisy image')
        self.ax[2].axis('off')
        
        plt.tight_layout()
        plt.show()

