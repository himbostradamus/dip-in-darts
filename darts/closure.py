#   __  __           _      _   ______          _             _   _             
#  |  \/  |         | |    | | |  ____|        | |           | | (_)            
#  | \  / | ___   __| | ___| | | |____   ____ _| |_   _  __ _| |_ _  ___  _ __  
#  | |\/| |/ _ \ / _` |/ _ \ | |  __\ \ / / _` | | | | |/ _` | __| |/ _ \| '_ \ 
#  | |  | | (_) | (_| |  __/ | | |___\ V / (_| | | |_| | (_| | |_| | (_) | | | |
#  |_|  |_|\___/ \__,_|\___|_| |______\_/ \__,_|_|\__,_|\__,_|\__|_|\___/|_| |_|
#    

import skimage
import numpy as np
from torch import cuda, optim, tensor, zeros_like
from torch import device as torch_device
from torch.nn import L1Loss, MSELoss

from .common_utils import *
from .phantom import generate_phantom, phantom_to_torch
from .noises import add_selected_noise


class EarlyStop():
    def __init__(self, size, patience):
        self.patience = patience
        self.wait_count = 0
        self.best_score = float('inf')
        self.best_epoch = 0
        self.img_collection = []
        self.stop = False
        self.size = size

    def check_stop(self, current, cur_epoch):
      #stop when variance doesn't decrease for consecutive P(patience) times
        if current < self.best_score:
            self.best_score = current
            self.best_epoch = cur_epoch
            self.wait_count = 0
            should_stop = False
        else:
            self.wait_count += 1
            should_stop = self.wait_count >= self.patience
        return should_stop

    def update_img_collection(self, cur_img):
        self.img_collection.append(cur_img)
        if len(self.img_collection) > self.size:
            self.img_collection.pop(0)

    def get_img_collection(self):
        return self.img_collection

def MSE(x1, x2):
    return ((x1 - x2) ** 2).sum() / x1.size

def MAE(x1, x2):
    return (np.abs(x1 - x2)).sum() / x1.size

def evaluate_model(model_cls):
    
    dtype = cuda.FloatTensor
    buffer_size = 100
    patience = 1000
    lr = 0.01
    num_iter = 2

    device = torch_device('cuda' if cuda.is_available() else "cpu")

    img_np = generate_phantom(resolution=6)
    img = phantom_to_torch(img_np)
    img_noisy_np = get_noise(img, spatial_size=(img.size[3], img.size[2]), noise_type="gaussian").to(device)
    img_noisy_torch = np_to_torch(img_noisy_np).to(device)
    net_input = img_noisy_torch.clone()


    # Add synthetic noise
    net = model_cls().to(device)
    net = net.type(dtype)

    # Loss
    criterion = MSELoss().type(dtype).to(device)

    # Optimizer

    p = get_params('net', net, net_input)  # network parameters to be optimized
    optimizer = optim.Adam(p, lr=lr)

    # Optimize

    # reg_noise_std = 1./30. 
    reg_noise_std = tensor(1./30.).type(dtype).to(device)
    show_every = 1
    loss_history = []
    psnr_history = []
    ssim_history = []
    variance_history = []
    x_axis = []
    earlystop = EarlyStop(size=buffer_size,patience=patience)
    def closure(iterator):
        #DIP
        net_input_perturbed = net_input + zeros_like(net_input).normal_(std=reg_noise_std)
        r_img_torch = net(net_input_perturbed)
        total_loss = criterion(r_img_torch, img_noisy_torch)
        total_loss.backward()
        loss_history.append(total_loss.item())
        if iterator % show_every == 0:
            # evaluate recovered image (PSNR, SSIM)
            r_img_np = torch_to_np(r_img_torch)
            psnr = skimage.metrics.peak_signal_noise_ratio(img_np, r_img_np)
            temp_img_np = np.transpose(img_np,(1,2,0))
            temp_r_img_np = np.transpose(r_img_np,(1,2,0))
            data_range = temp_img_np.max() - temp_img_np.min()
            ssim = skimage.metrics.structural_similarity(temp_img_np, temp_r_img_np, multichannel=True, win_size=7, channel_axis=-1, data_range=data_range)
            psnr_history.append(psnr)
            ssim_history.append(ssim)

            #variance hisotry
            r_img_np = r_img_np.reshape(-1)
            earlystop.update_img_collection(r_img_np)
            img_collection = earlystop.get_img_collection()
            if len(img_collection) == buffer_size:
                ave_img = np.mean(img_collection,axis = 0)
                variance = []
                for tmp in img_collection:
                    variance.append(MSE(ave_img, tmp))
                cur_var = np.mean(variance)
                cur_epoch = iterator
                variance_history.append(cur_var)
                x_axis.append(cur_epoch)
                if earlystop.stop == False:
                    earlystop.stop = earlystop.check_stop(cur_var, cur_epoch)
        return total_loss
        
    for iterator in range(num_iter):
        optimizer.zero_grad()
        closure(iterator)
        optimizer.step()