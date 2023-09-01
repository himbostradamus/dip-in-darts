import numpy as np
import torch

import pytorch_lightning as pl

from nni import trace, report_intermediate_result, report_final_result
import nni.retiarii.nn.pytorch as nn
from nni.retiarii.evaluator.pytorch import LightningModule
from nni.retiarii.evaluator.pytorch.lightning import DataLoader
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from torch import optim, tensor, zeros_like
from typing import Any

from darts.common_utils import *
from darts.early_stop import MSE
from darts.noises import add_selected_noise
from darts.phantom import generate_phantom

from torch.utils.data import Dataset


@trace
class SingleImageDataset(Dataset):
    def __init__(self, image, num_iter):
        self.image = image
        self.num_iter = num_iter

    def __len__(self):
        return self.num_iter

    def __getitem__(self, index):
        # Always return the same image (and maybe a noise tensor or other information if necessary??)
        return self.image

@trace
class LightningEvalSearch(LightningModule):
    def __init__(self, phantom=None, buffer_size=100, num_iter=1,
                lr=0.01, noise_type='gaussian', noise_factor=0.15, resolution=6, 
                n_channels=1, reg_noise_std_val=1./30.,
                buffer_no_lr_schuler=700, patience=100,
                ):
        super().__init__()

        # input
        self.phantom = phantom

        # Loss
        # self.criterion = nn.MSELoss().to(self.device)
        self.criterion = nn.BCEWithLogitsLoss().to(self.device)
        
        # Hyperparameters / Inputs
        self.buffer_size = buffer_size
        self.num_iter = num_iter
        self.lr = lr
        self.noise_type = noise_type
        self.noise_factor = noise_factor
        self.resolution = resolution
        self.n_channels = n_channels
        self.reg_noise_std = tensor(reg_noise_std_val)

        self.buffer_no_lr_schuler = buffer_no_lr_schuler
        self.patience = patience

        # adjusting input
        if self.phantom is None:
            self.img_np, _, _, self.img_noisy_torch = self.preprocess_image(self.resolution, self.noise_type, self.noise_factor)
        else:
            self.img_np, _, _, self.img_noisy_torch = self.preprocess_image(self.resolution, self.noise_type, self.noise_factor, input_img_np=self.phantom)
        self.net_input = get_noise(input_depth=1, spatial_size=self.img_np.shape[1], noise_type=self.noise_type)
        
        # History and early stopper
        self.loss_history = []
        self.variance_history = []
        self.img_collection = []
        
    def forward(self, net_input):
        net_input_perturbed = net_input + zeros_like(net_input).normal_(std=self.reg_noise_std)
        return self.model(net_input_perturbed)
    
    def training_step(self, batch, batch_idx):
        """
        Deep Image Prior

        training here follows closely from the following two repos: 
            - the deep image prior repo
            - a DIP early stopping repo (Lighting has early stopping functionality so this blends the two)
        """        
        self.counter += 1

        r_img_torch = self.forward(self.net_input)
        r_img_np = torch_to_np(r_img_torch)

        # update loss and loss history
        total_loss = self.criterion(r_img_torch, self.img_noisy_torch)
        self.latest_loss = total_loss.item()
        self.loss_history.append(total_loss.item())
        # self.logger.log_metrics({'loss': total_loss.item()})

        # update img collection
        r_img_np = r_img_np.reshape(-1)
        self.update_img_collection(r_img_np)
        img_collection = self.get_img_collection()

        # if len(img_collection) == self.buffer_size:
        if len(img_collection) >= self.buffer_size // 2:

            # update variance and var history
            ave_img = np.mean(img_collection, axis=0)
            variance = [MSE(ave_img, tmp) for tmp in img_collection]
            self.cur_var = np.mean(variance)
            self.variance_history.append(self.cur_var)

            # update log
            #self.latest_loss = total_loss.item()
            self.log('variance', self.cur_var)
            self.log('loss', self.latest_loss)

            # Using global_step to count iterations
            if self.counter % 10 == 0:
                report_intermediate_result({'Iteration':self.counter,'variance':self.cur_var, 'loss': self.latest_loss})
        else:
            #self.latest_loss = total_loss.item()
            self.log('loss', self.latest_loss)
            # log a fake variance to fool the early stopper and lr scheduler
            self.log('variance', 1-self.counter/1000000)
            if self.counter % 10 == 0:
                report_intermediate_result({'Iteration':self.counter,'variance': "iterations <= buffer_size // 2", 'loss': total_loss.item()})
        
        if self.counter % 200 == 0:
            self.plot_progress()

        return {"loss": total_loss}
    
    def validation_step(self, batch, batch_idx):
        # your validation logic here
        return {'loss': self.latest_loss, 'variance': self.cur_var} if self.counter >= self.buffer_size else {'loss': self.latest_loss, 'variance': 1-self.counter/1000000}


    def configure_optimizers(self):
        """
        Basic Adam Optimizer
        LR Scheduler: ReduceLROnPlateau
        1 Parameter Group
        """
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        # self.buffer_no_lr_schuler = 700

        self.monitor = 'variance'
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=self.patience, verbose=True)
    
        # Reduce learning rate when a metric has stopped improving
        # Here 'min' means that the metric should decrease for the lr to be reduced
        scheduler = {
            'scheduler': self.scheduler,
            'monitor': self.monitor, # probably want this to be variance eventually
            'interval': 'step',
            'strict': True,
            'frequency': 1,
        }
        
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
        # return optimizer
        
    def lr_scheduler_step(self, *args, **kwargs):
        # needed to modify
        # one-shot methods interfere with this for some reason
        

        scheduler = kwargs.get('scheduler', args[0] if args else None)
        metric = kwargs.get('metric', args[1] if len(args) > 1 else None)
        metric = self.trainer.logged_metrics[self.monitor]

        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau) and metric is not None:
            if self.counter >= self.buffer_no_lr_schuler:
                scheduler.step(metric) # self.cur_var
        else:
            scheduler.step()
    
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

    def set_model(self, model):
        # This will be called after __init__ and will set the candidate model
        # needed for NAS but not for a standard training loop
        self.model = model
    
    def train_dataloader(self):
        """
        Dummy DataLoader that returns nothing but makes PyTorch Lightning's training loop work
        """
        dataset = SingleImageDataset(self.phantom, self.num_iter)
        return DataLoader(dataset, batch_size=1)
    
    def on_train_start(self):
        """
        Move all tensors to the GPU to begin training
        """
        self.model.to(self.device)
        self.net_input = self.net_input.to(self.device)
        self.img_noisy_torch = self.img_noisy_torch.to(self.device)
        self.reg_noise_std = self.reg_noise_std.to(self.device)
        self.counter = 0

    def on_train_end(self, **kwargs: Any):
        """
        Report final metrics and display the results
        """
        # final log
        report_final_result({'variance': self.cur_var, 'loss': self.latest_loss})

        # # plot images to see results
        self.plot_progress()


    # def test_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT | None:
    #     pass

    def val_dataloader(self):
        """
        Dummy DataLoader for validation.
        """
        dataset = SingleImageDataset(self.phantom, self.num_iter)
        return DataLoader(dataset, batch_size=1)
    
    # def validation_step(self, trainer, *args: Any, **kwargs: Any) -> STEP_OUTPUT | None:
    #     # if self.buffer_size == len(self.img_collection):
    #     #     self._run_early_stopping_check(trainer)
    #     pass

    def update_img_collection(self, cur_img):
        self.img_collection.append(cur_img)
        if len(self.img_collection) > self.buffer_size:
            self.img_collection.pop(0)

    def get_img_collection(self):
        return self.img_collection

    def preprocess_image(self, resolution, noise_type, noise_factor, input_img_np=None):
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
    
    def plot_progress(self):
        denoised_img = self.forward(self.net_input).detach().cpu().squeeze().numpy()
        
        _, ax = plt.subplots(1, 3, figsize=(10, 5))

        ax[0].imshow(self.img_np.squeeze(), cmap='gray')
        ax[0].set_title("Original Image")
        ax[0].axis('off')

        ax[1].imshow(denoised_img, cmap='gray')
        ax[1].set_title("Denoised Image")
        ax[1].axis('off')

        ax[2].imshow(self.img_noisy_torch.detach().cpu().squeeze().numpy(), cmap='gray')
        ax[2].set_title("Noisy Image")
        ax[2].axis('off')

        plt.tight_layout()
        plt.show()

@trace
class _EarlyStopping(EarlyStopping, pl.Callback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        

# early stopper
custom_early_stop_callback = _EarlyStopping(
                        monitor="variance", 
                        mode="min", 
                        patience=6, 
                        verbose=True,
                        min_delta=0
                        )