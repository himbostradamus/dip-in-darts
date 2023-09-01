
import nni
import numpy as np
import skimage
import torch
import nni.retiarii.nn.pytorch as nn

from nni.retiarii import model_wrapper

from skimage.metrics import peak_signal_noise_ratio as skimage_psnr

from torch import cuda, optim, tensor, zeros_like
from torch import device as torch_device

from .common_utils import *
from .early_stop import EarlyStop, MSE, MAE
from .noises import add_selected_noise
from .phantom import generate_phantom, phantom_to_torch


def psnr(image_true, image_test):
    # Convert PyTorch tensors to NumPy arrays
    if torch.is_tensor(image_true):
        image_true = image_true.detach().cpu().numpy()
    if torch.is_tensor(image_test):
        image_test = image_test.detach().cpu().numpy()
    return skimage_psnr(image_true, image_test)

def deep_image_prior_denoising(model, noisy_img, clean_img, device, optimizer, iterations=250):
    model.train()
    for iteration in range(iterations):
        optimizer.zero_grad()
        output = model(torch.randn(noisy_img.shape).to(device))
        loss = nn.MSELoss()(output, noisy_img)
        loss.backward()
        optimizer.step()
        if iteration % 25 == 0:
            # Calculate PSNR
            with torch.no_grad():
                denoised_output = model(noisy_img)
                print("Shape of denoised_output:", denoised_output.shape)
                print("Shape of clean_img:", clean_img.shape)
                # psnr_value = psnr(clean_img.squeeze(1), denoised_output)
                psnr_value = psnr(clean_img, denoised_output)
            print('Iteration: {}\tLoss: {:.6f}\tPSNR: {:.6f} dB'.format(iteration, loss.item(), psnr_value))
            nni.report_intermediate_result(psnr_value)
    return output

def main_evaluation(model_cls):
    # change to simple_evaluation

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # Instantiate model
    model = model_cls().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Generate the phantom image and convert the numpy img to torch tensor
    img_numpy = generate_phantom(resolution=6)
    img = phantom_to_torch(img_numpy)
    noisy_img = add_selected_noise(img, noise_type="gaussian").to(device)

    # Denoise the image for a set number of iterations
    # denoised_img = deep_image_prior_denoising(model, noisy_img, img.unsqueeze(0).to(device), device, optimizer)
    denoised_img = deep_image_prior_denoising(model, noisy_img, img.to(device), device, optimizer)

    # # Evaluate the PSNR of the denoised image
    # psnr_value = psnr(img.unsqueeze(0).to(device), denoised_img)
    psnr_value = psnr(img.to(device), denoised_img)

    print('PSNR: {:.6f} dB'.format(psnr_value))

    # Report final PSNR to NNI
    nni.report_final_result(psnr_value.item())

def preprocess_image(resolution, noise_type, noise_factor, input_img_np=None):
    """
    Generates an image, adds noise, and converts it to both numpy and torch tensors.

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
        raw_img_np = input_img_np.copy()
        
    img_np = raw_img_np.copy() # 1x64x64 np array
    img_torch = torch.tensor(raw_img_np, dtype=torch.float32).unsqueeze(0) # 1x1x64x64 torch tensor
    img_noisy_torch = add_selected_noise(img_torch, noise_type=noise_type, noise_factor=noise_factor) # 1x1x64x64 torch tensor
    img_noisy_np = img_noisy_torch.squeeze(0).numpy() # 1x64x64 np array
    
    return img_np, img_noisy_np, img_torch, img_noisy_torch


def main_evaluation_with_closure(model_cls):
    device = torch_device('cuda' if cuda.is_available() else "cpu")
    dtype = cuda.FloatTensor if cuda.is_available() else torch.FloatTensor

    buffer_size = 100
    patience = 600
    num_iter = 1200
    show_every = 1
    lr = 0.00005

    reg_noise_std = tensor(1./30.).type(dtype).to(device)
    noise_type = 'gaussian'
    noise_factor = 0.1
    resolution= 6
    n_channels = 1

    img_np, _, _, img_noisy_torch = preprocess_image(resolution, noise_type, noise_factor)
    img_noisy_torch = img_noisy_torch.to(device)
    net_input = get_noise(input_depth=1, spatial_size=img_np.shape[1], noise_type=noise_type).type(dtype).to(device)

    net = model_cls().to(device)
    net = net.type(dtype)

    # Loss
    criterion = nn.MSELoss().type(dtype).to(device)

    # Optimizer
    p = get_params('net', net, net_input)  # network parameters to be optimized
    optimizer = optim.Adam(p, lr=lr)

    # Optimize

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
            if n_channels == 1:
                multichannel = False
            else:
                multichannel = True
            ssim = skimage.metrics.structural_similarity(temp_img_np, temp_r_img_np, multichannel=multichannel, win_size=7, channel_axis=-1, data_range=data_range)
            psnr_history.append(psnr)
            ssim_history.append(ssim)
            
            #variance hisotry
            r_img_np = r_img_np.reshape(-1)
            earlystop.update_img_collection(r_img_np)
            img_collection = earlystop.get_img_collection()
            if iterator % (show_every*10) == 0:
                print(f'Iteration %05d    Loss %.4f' % (iterator, total_loss.item()) + '    PSNR %.4f' % (psnr) + '    SSIM %.4f' % (ssim))
                nni.report_intermediate_result(psnr)
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
        if earlystop.stop:
            # Report final PSNR to NNI
            nni.report_final_result(psnr)
            return "STOP"
        return total_loss, psnr
        
    for iterator in range(num_iter):
        optimizer.zero_grad()
        early_stop, psnr = closure(iterator)
        optimizer.step()

        if early_stop == "STOP":
            print("Early stopping triggered.")
            break
    
    if earlystop.stop != "STOP":
        nni.report_final_result(psnr)
    