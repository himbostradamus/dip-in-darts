from nni import trace, report_intermediate_result, report_final_result
from nni.retiarii.evaluator.pytorch import LightningModule
from nni.retiarii.evaluator.pytorch.lightning import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from skimage.metrics import peak_signal_noise_ratio as compare_psnr

import torch
from torch.optim import Optimizer
from torch.utils.data import Dataset
from torch import optim, tensor

from typing import Any

from sgld.utils import *
from noises import add_selected_noise
from phantom import generate_phantom

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
                n_channels=1, reg_noise_std_val=1./30., burnin_iter=1800, MCMC_iter=50,
                model_cls=None, show_every=200
                ):
        super().__init__()
        torch.autograd.set_detect_anomaly(True)

        self.model_cls = model_cls
        self.total_loss = 10

        self.checkpoint_callback = ModelCheckpoint(
            dirpath="~/nas-for-dip/checkpoints", 
            filename="checkpoint_{epoch}",
            save_top_k=50,  # save the last two epochs for backtracking
            monitor="psrn_noisy", 
            mode="max"
        )

        self.reg_noise_std = tensor(reg_noise_std_val)
        self.learning_rate = lr
        self.roll_back = True # to prevent numerical issues
        self.num_iter = num_iter # max iterations
        self.burnin_iter = burnin_iter # burn-in iteration for SGLD
        self.weight_decay = 5e-8
        self.show_every =  show_every
        self.criteria = torch.nn.MSELoss().type(dtype) # loss

        
        self.sgld_mean_each = 0
        self.sgld_psnr_mean_list = []
        self.MCMC_iter = MCMC_iter

        self.sgld_mean = 0
        self.last_net = None
        self.psnr_noisy_last = 0
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
        if self.model_cls is not None:
            self.model = self.model_cls
        self.model = model

    def configure_optimizers(self):
        """
        Basic Adam Optimizer
        LR Scheduler: ReduceLROnPlateau
        1 Parameter Group
        """
        print('Starting optimization with SGLD')
        #optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        optimizer = SGLD(self.model.parameters(), lr=self.learning_rate, num_burn_in_steps=self.burnin_iter)
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
        # self.total_loss.backward()
        self.latest_loss = self.total_loss.item()
        self.log('loss', self.latest_loss)
        out_np = out.detach().cpu().numpy()[0]

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

        if self.iteration_counter % 10 == 0 and self.i > self.burnin_iter:
            report_intermediate_result({'iteration': self.iteration_counter ,'loss': self.latest_loss, 'sample count': self.i - self.burnin_iter, 'psnr_sgld_last': self.sgld_psnr_mean_list[-1]})
        elif self.iteration_counter % 10 == 0:
            report_intermediate_result({'iteration': self.iteration_counter ,'loss': self.latest_loss, 'psnr_noisy': self.psnr_noisy})
            # print('\rIteration: {}, Loss: {}, PSNR Noisy: {}'.format(self.iteration_counter, self.latest_loss, self.psnr_noisy), end='')

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

        # print(f"entering forward pass: {self.iteration_counter}")
        r_img_torch = self.forward(self.net_input)
        # print(f"completed forward pass: {self.iteration_counter}")

        # print(f"entering closure: {self.iteration_counter}")
        self.closure(r_img_torch)
        # print(f"completed closure: {self.iteration_counter}")

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

    # def val_dataloader(self):
    #     return self.common_dataloader()
    
    # def validation_step(self, batch, batch_idx):
    #     # your validation logic here
    #     return {'loss': self.total_loss}
    
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

class SGLD(Optimizer):
    """ Stochastic Gradient Langevin Dynamics Sampler with preconditioning.
        Optimization variable is viewed as a posterior sample under Stochastic
        Gradient Langevin Dynamics with noise rescaled in each dimension
        according to RMSProp.

        https://pysgmcmc.readthedocs.io/en/pytorch/_modules/pysgmcmc/optimizers/sgld.html
    """
    def __init__(self,
                 params,
                 lr=1e-2,
                 precondition_decay_rate=0.95,
                 num_pseudo_batches=1,
                 num_burn_in_steps=3000,
                 diagonal_bias=1e-8) -> None:
        """ Set up a SGLD Optimizer.

        Parameters
        ----------
        params : iterable
            Parameters serving as optimization variable.
        lr : float, optional
            Base learning rate for this optimizer.
            Must be tuned to the specific function being minimized.
            Default: `1e-2`.
        precondition_decay_rate : float, optional
            Exponential decay rate of the rescaling of the preconditioner (RMSprop).
            Should be smaller than but nearly `1` to approximate sampling from the posterior.
            Default: `0.95`
        num_pseudo_batches : int, optional
            Effective number of minibatches in the data set.
            Trades off noise and prior with the SGD likelihood term.
            Note: Assumes loss is taken as mean over a minibatch.
            Otherwise, if the sum was taken, divide this number by the batch size.
            Default: `1`.
        num_burn_in_steps : int, optional
            Number of iterations to collect gradient statistics to update the
            preconditioner before starting to draw noisy samples.
            Default: `3000`.
        diagonal_bias : float, optional
            Term added to the diagonal of the preconditioner to prevent it from
            degenerating.
            Default: `1e-8`.

        """
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if num_burn_in_steps < 0:
            raise ValueError("Invalid num_burn_in_steps: {}".format(num_burn_in_steps))

        defaults = dict(
            lr=lr, precondition_decay_rate=precondition_decay_rate,
            num_pseudo_batches=num_pseudo_batches,
            num_burn_in_steps=num_burn_in_steps,
            diagonal_bias=1e-8,
        )
        super().__init__(params, defaults)


    def step(self, closure=None):
        loss = None

        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for parameter in group["params"]:

                if parameter.grad is None:
                    continue

                state = self.state[parameter]
                lr = group["lr"]
                num_pseudo_batches = group["num_pseudo_batches"]
                precondition_decay_rate = group["precondition_decay_rate"]
                gradient = parameter.grad.data

                #  State initialization {{{ #

                if len(state) == 0:
                    state["iteration"] = 0
                    state["momentum"] = torch.ones_like(parameter)

                #  }}} State initialization #

                state["iteration"] += 1

                momentum = state["momentum"]

                #  Momentum update {{{ #
                momentum.add_(
                    (1.0 - precondition_decay_rate) * ((gradient ** 2) - momentum)
                )
                #  }}} Momentum update #

                if state["iteration"] > group["num_burn_in_steps"]:
                    sigma = 1. / torch.sqrt(torch.tensor(lr))
                else:
                    sigma = torch.zeros_like(parameter)

                preconditioner = (
                    1. / torch.sqrt(momentum + group["diagonal_bias"])
                )

                scaled_grad = (
                    0.5 * preconditioner * gradient * num_pseudo_batches +
                    torch.normal(
                        mean=torch.zeros_like(gradient),
                        std=torch.ones_like(gradient)
                    ) * sigma * torch.sqrt(preconditioner)
                )

                parameter.data.add_(-lr * scaled_grad)

        return loss

### old backtracking logic ###

        # Backtracking logic
        # if self.roll_back and self.iteration_counter % self.show_every == 0:
        #     if self.psnr_noisy - self.psnr_noisy_last < -5:
        #         print(f'itearation: {self.iteration_counter} -- Falling back to previous checkpoint.')
                
        #         # # Load checkpoint
        #         # self.load_from_checkpoint(self.checkpoint_callback.best_model_path)

        #         # Load checkpoint
        #         checkpoint = torch.load(self.checkpoint_callback.best_model_path)
        #         self.model.load_state_dict(checkpoint['state_dict'])
                
        #         print(f"Resumed training from checkpoint {self.checkpoint_callback.best_model_path}")
                
        #     self.psnr_noisy_last = self.psnr_noisy

        #    self.i += 1