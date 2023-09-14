import matplotlib.pyplot as plt
import numpy as np

from nni import trace, report_intermediate_result, report_final_result
from nni.retiarii.evaluator.pytorch import LightningModule
from nni.retiarii.evaluator.pytorch.lightning import DataLoader

from skimage.metrics import peak_signal_noise_ratio as compare_psnr

import torch
from torch.optim import Optimizer
from torch import tensor

from typing import Any

from .utils.common_utils import get_noise
from .optimizer.SingleImageDataset import SingleImageDataset

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
dtype = torch.cuda.FloatTensor

@trace
class SGLDES(LightningModule):
    def __init__(self, 
                 phantom=None,
                 phantom_noisy=None,
                 
                 learning_rate=0.01,
                 buffer_size=100,
                 patience=1000,
                 weight_decay = 5e-8,

                 MCMC_iter=50, 
                 show_every=200,
                 report_every=25,

                 model_cls=None,
                 HPO=False,
                 NAS=False,
                 OneShot=False,
                 SGLD_regularize=True,
                 ES=True,
                 switch=None,
                 plotting=True
                ):
        super().__init__()
        self.automatic_optimization = True
        self.HPO = HPO
        self.NAS = NAS
        self.OneShot = OneShot
        self.SGLD_regularize = SGLD_regularize
        self.switch = switch
        self.ES = ES
        self.plotting = plotting

        # network input
        self.input_depth = 1

        if NAS and OneShot:
            self.model_cls = model_cls
        
        if not NAS:
            if model_cls is None:
                model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet', in_channels=1, out_channels=1, init_features=64, pretrained=False)
            self.model_cls = model
            self.model = model

        # loss
        self.criteria = torch.nn.MSELoss().type(dtype)

        # "Early Stopper" Trigger
        # modifying an early stopper for DIP to determin more programatically when the SGLD burn in period is finished
        self.reg_noise_std = tensor(1./30.)
        self.learning_rate = learning_rate
        self.roll_back = True
        self.weight_decay = weight_decay
        self.show_every =  show_every
        self.report_every = report_every

        # SGLD Optimization
        # SGLD takes the average of every n samples after the burn in period as the final reconstruction
        self.sgld_mean_each = 0
        self.MCMC_iter = MCMC_iter # here is the n from the above comment
        self.sgld_mean = 0
        self.param_noise_sigma = 2



        
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
            - Implementing this would greatly affect the training step
                - But could it work?? :`( I couldn't figure it out
        """
        return torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

    def set_model(self, model):
        if self.NAS and self.OneShot:
            if self.model_cls is not None:
                self.model = self.model_cls
            self.model = model
        
        if self.NAS and not self.OneShot:
            self.model = model()
        
        if not self.NAS:
            pass

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

        # bon voyage
        if self.plotting:
            self.plot_progress()

    def forward(self, net_input_saved):
        if self.reg_noise_std > 0:
            self.net_input = self.net_input_saved + (self.noise.normal_() * self.reg_noise_std)
            return self.model(self.net_input)
        else:
            return self.model(net_input_saved)

    def update_stop(self,out_np):
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

    def sgld_closure_calc(self,out_np):
        ##########################################
        ### Logging and SGLD mean collection #####
        ##########################################
        if self.burnin_over and np.mod(self.i, self.MCMC_iter) == 0:
            self.sgld_mean += out_np
            self.sample_count += 1.
            self.sgld_mean_psnr = compare_psnr(self.img_np, self.sgld_mean / self.sample_count)

        if self.burnin_over:
            self.burnin_iter+=1
            self.sgld_mean_each += out_np
            self.sgld_mean_tmp = self.sgld_mean_each / self.burnin_iter
            self.sgld_mean_psnr_each = compare_psnr(self.img_np, self.sgld_mean_tmp)

            if self.i % self.report_every == 0:
                if not self.HPO:
                    report_intermediate_result({
                        'iteration': self.i,
                        'loss': round(self.latest_loss,5),
                        'psnr_gt': round(self.psnr_gt,5),
                        'psnr': round(self.sgld_mean_psnr_each,5)
                        })
        
        elif self.cur_var is not None and not self.burnin_over:
            if self.i % self.report_every == 0:
                if not self.HPO:
                    report_intermediate_result({
                        'iteration': self.i,
                        'loss': round(self.latest_loss,5),
                        'psnr_gt': round(self.psnr_gt,5),
                        'var': round(self.cur_var,5)
                        })

        else:
            if self.i % self.report_every == 0:
                if not self.HPO:
                    report_intermediate_result({
                        'iteration': self.i,
                        'loss': round(self.latest_loss,5),
                        'psnr_gt': round(self.psnr_gt,5),
                        })

        if self.i % self.report_every == 0 and self.HPO:
            report_intermediate_result(round(self.psnr_gt,5))

    def closure(self):
        out = self.forward(self.net_input)

        # compute loss
        self.total_loss = self.criteria(out, self.img_noisy_torch)
        self.latest_loss = self.total_loss.item()
        out_np = out.detach().cpu().numpy()[0]

        # compute PSNR
        self.psnr_gt = compare_psnr(self.img_np, out_np)

        # early burn in termination criteria
        if not self.burnin_over and self.ES:
            self.update_stop(out_np)

        # SGLD mean calculation and logging
        if self.SGLD_regularize:
            self.sgld_closure_calc(out_np)

        # Non SGLD mean logging
        elif self.i % self.report_every == 0 and not self.HPO:
            report_intermediate_result({
                'iteration': self.i,
                'loss': round(self.latest_loss,5),
                'psnr_gt': round(self.psnr_gt,5),
                'var': round(self.cur_var,5) if self.ES and self.cur_var is not None else 'Pre-Burnin'
                })
        elif self.i % self.report_every == 0 and self.HPO:
            report_intermediate_result(round(self.psnr_gt,5))

        self.i += 1
        return self.total_loss

    # Define hook 
    def on_train_batch_start(self, batch, batch_idx):
        if self.NAS and self.OneShot:
            optimizer = self.optimizers()[0]
            optimizer.zero_grad()

    def training_step(self, batch: Any, batch_idx: int) -> Any:
        """
        Oh the places you'll go
        """
        loss = self.closure()
        return {"loss": loss}

    def add_noise(self, net):
        """
        Add noise to the network parameters
        This is the critical part of SGLD
        """
        for n in [x for x in net.parameters() if len(x.size()) == 4]:
            noise = torch.randn(n.size())*self.param_noise_sigma*self.learning_rate
            noise = noise.type(self.dtype).to(self.device)
            n.data = n.data + noise

    def on_train_batch_end(self, outputs, batch, batch_idx, *args, **kwargs):
        """
        Add noise for SGLD
        """
        optimizer = self.optimizers()
        if isinstance(optimizer, torch.optim.Adam) and self.SGLD_regularize:
            self.add_noise(self.model)

        if self.i % self.show_every == 0 and not self.HPO:
            if self.plotting:
                self.plot_progress()

        if self.switch is not None:
            if self.i >= self.switch:
                self.SGLD_regularize = True
        
        if self.burnin_over and self.ES and not self.SGLD_regularize:
            print(f'Early stopping after {self.i} iterations')
            self.trainer.should_stop = True

    def on_train_end(self, **kwargs: Any):
        """
        Report final metrics and display the results
        """
        if not self.HPO:
            if self.plotting:
                self.plot_progress()
            if self.sample_count != 0 and self.SGLD_regularize:
                print(f"Final SGLD mean PSNR: {round(self.sgld_mean_psnr,5)}")
                report_final_result(round(self.sgld_mean_psnr,5))
            else:
                print(f"Final PSNR: {round(self.psnr_gt,5)}")
                report_final_result(round(self.psnr_gt,5))            
        if self.HPO and self.sample_count != 0 and self.SGLD_regularize:
            report_final_result(round(self.sgld_mean_psnr,5))
        if self.HPO and self.sample_count == 0:
            report_final_result(round(self.psnr_gt,5))

    def common_dataloader(self):
        # dataset = SingleImageDataset(self.phantom, 1)
        # this should be net_input instead of phantom
        dataset = SingleImageDataset(self.img_noisy_np, 1)
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
            # self.show_every = self.MCMC_iter

    def update_img_collection(self, cur_img):
        self.img_collection.append(cur_img)
        if len(self.img_collection) > self.buffer_size:
            self.img_collection.pop(0)

    def get_img_collection(self):
        return self.img_collection

    def MSE(self, x1, x2):
        return ((x1 - x2) ** 2).sum() / x1.size
        
    def plot_progress(self):
        if self.sample_count == 0:
            denoised_img = self.forward(self.net_input).detach().cpu().squeeze().numpy()
            label = "Denoised Image"
        else:
            denoised_img = self.sgld_mean_each / self.burnin_iter
            denoised_img = np.squeeze(denoised_img)
            label = "SGLD Mean"

        _, self.ax = plt.subplots(1, 3, figsize=(10, 5))

        self.ax[0].imshow(self.img_np.squeeze(), cmap='gray')
        self.ax[0].set_title("Original Image")
        self.ax[0].axis('off')

        self.ax[1].imshow(denoised_img, cmap='gray')
        self.ax[1].set_title(label)
        self.ax[1].axis('off')

        self.ax[2].imshow(self.img_noisy_torch.detach().cpu().squeeze().numpy(), cmap='gray')
        self.ax[2].set_title("Noisy Image")
        self.ax[2].axis('off')

        plt.tight_layout()
        plt.show()