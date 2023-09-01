from nni import trace, report_intermediate_result, report_final_result
from nni.retiarii.evaluator.pytorch import LightningModule
from nni.retiarii.evaluator.pytorch.lightning import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from skimage.metrics import peak_signal_noise_ratio as compare_psnr

import torch
from torch.utils.data import Dataset
from torch import optim, tensor

from typing import Any

from darts.common_utils import *
from darts.noises import add_selected_noise
from darts.phantom import generate_phantom

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
dtype = torch.cuda.FloatTensor

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
class LightningEvalSearchSGLD(LightningModule):
    def __init__(self, phantom=None, num_iter=1,
                lr=0.01, noise_type='gaussian', noise_factor=0.15, resolution=6, 
                n_channels=1, reg_noise_std_val=1./30.
                ):
        super().__init__()

        torch.autograd.set_detect_anomaly(True)

        self.checkpoint_callback = ModelCheckpoint(
            dirpath="./checkpoints", filename="checkpoint_{epoch}",
            save_top_k=2,  # save the last two epochs for backtracking
            monitor="psrn_noisy", mode="max"
)

        self.reg_noise_std = tensor(reg_noise_std_val)
        self.learning_rate = lr
        self.roll_back = True # to prevent numerical issues
        self.num_iter = num_iter # max iterations
        self.burnin_iter = 7000 # burn-in iteration for SGLD
        self.weight_decay = 5e-8
        self.show_every =  500
        self.criteria = torch.nn.MSELoss().type(dtype) # loss

        self.sgld_mean = 0
        self.last_net = None
        self.psnr_noisy_last = 0
        self.MCMC_iter = 50
        self.param_noise_sigma = 2

        self.noise_type=noise_type
        self.noise_factor=noise_factor
        self.resolution = resolution
        self.phantom = generate_phantom(resolution=resolution)

        if phantom is None:
            self.img_np, self.img_noisy_np, _, self.img_noisy_torch = self.preprocess_image(self.resolution, self.noise_type, self.noise_factor)
        else:
            self.img_np, self.img_noisy_np, _, self.img_noisy_torch = self.preprocess_image(self.resolution, self.noise_type, self.noise_factor, input_img_np=self.phantom)
        self.net_input = get_noise(input_depth=n_channels, spatial_size=self.img_np.shape[1], noise_type=self.noise_type)
    
    def set_model(self, model):
        # This will be called after __init__ and will set the candidate model
        # needed for NAS but not for a standard training loop
        self.model = model

    def configure_optimizers(self):
        """
        Basic Adam Optimizer
        LR Scheduler: ReduceLROnPlateau
        1 Parameter Group
        """
        print('Starting optimization with SGLD')
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
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
        torch.autograd.set_detect_anomaly(True)
        out = r_img_torch
        self.total_loss = self.criteria(out, self.img_noisy_torch)
        # self.total_loss.backward()
        self.latest_loss = self.total_loss.item()
        self.log('loss', self.latest_loss)
        out_np = out.detach().cpu().numpy()[0]

        self.psnr_noisy = compare_psnr(self.img_noisy_np, out_np)
        self.log("psrn_noisy", self.psnr_noisy)

        # Backtracking logic
        if self.roll_back and self.iteration_counter % self.show_every == 0:
            if self.psnr_noisy - self.psnr_noisy_last < -5:
                print(f'itearation: {self.iteration_counter} -- Falling back to previous checkpoint.')
                
                # # Load checkpoint
                # self.load_from_checkpoint(self.checkpoint_callback.best_model_path)

                # Load checkpoint
                checkpoint = torch.load(self.checkpoint_callback.best_model_path)
                self.model.load_state_dict(checkpoint['state_dict'])
                
                print(f"Resumed training from checkpoint {self.checkpoint_callback.best_model_path}")
                
            self.psnr_noisy_last = self.psnr_noisy

            self.i += 1
        if self.iteration_counter % 25 == 0:
            report_intermediate_result({'iteration': self.iteration_counter ,'loss': self.latest_loss, 'psnr_noisy': self.psnr_noisy})

    def training_step(self, batch, batch_idx):
        """
        Deep Image Prior

        training here follows closely from the following two repos: 
            - the deep image prior repo
            - a DIP early stopping repo (Lighting has early stopping functionality so this blends the two)
        """      
        torch.autograd.set_detect_anomaly(True)
        self.iteration_counter += 1

        # print(f"entering forward pass: {self.iteration_counter}")
        r_img_torch = self.forward(self.net_input)
        # print(f"completed forward pass: {self.iteration_counter}")

        # print(f"entering closure: {self.iteration_counter}")
        self.closure(r_img_torch)
        # print(f"completed closure: {self.iteration_counter}")

        if self.iteration_counter % self.show_every == 0:
            self.plot_progress()
        return {"loss": self.total_loss}
    
    def on_after_step(self):
        """
        Add noise
        """
        print(f"entering add noise: {self.iteration_counter}")
        self.add_noise(self.model)
        print(f"completed add noise: {self.iteration_counter}")

    def on_train_end(self, **kwargs: Any):
        """
        Report final metrics and display the results
        """
        # final log
        report_final_result({'loss': self.latest_loss, 'psnr_noisy': self.psnr_noisy_last})

        # # plot images to see results
        self.plot_progress()

    def common_dataloader(self):
        dataset = SingleImageDataset(self.phantom, self.num_iter)
        return DataLoader(dataset, batch_size=1)

    def train_dataloader(self):
        return self.common_dataloader()

    def val_dataloader(self):
        return self.common_dataloader()
    
    def validation_step(self, batch, batch_idx):
        # your validation logic here
        torch.autograd.set_detect_anomaly(True)
        return {'loss': self.total_loss}
    
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
    
    def add_noise(self, net):
        for n in [x for x in net.parameters() if len(x.size()) == 4]:
            noise = torch.randn(n.size())*self.param_noise_sigma*self.learning_rate
            noise = noise.to(self.device)
            n.data = n.data + noise

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



### old backtracking ###
        # # Backtracking
        # if self.roll_back and self.i % self.show_every:
        #     if psrn_noisy - self.psrn_noisy_last < -5: 
        #         print('Falling back to previous checkpoint.')
        #         for new_param, net_param in zip(self.last_net, self.model.parameters()):
        #             net_param.detach().copy_(new_param.cuda())
        #         print("completed checkpoint loop")
        #         return self.total_loss*0
        #     else:
        #         self.last_net = [x.detach().cpu() for x in self.model.parameters()]
        #         self.psrn_noisy_last = psrn_noisy