import torch
import numpy as np

def add_gaussian_noise(img, noise_factor=0.15):
    img_copy = img.clone()
    noise = torch.randn_like(img_copy) * noise_factor
    noisy_img = img_copy + noise
    return torch.clamp(noisy_img, 0., 1.)

def add_salt_and_pepper_noise(img, salt_prob=0.01, pepper_prob=0.01):
    img_copy = img.clone()
    total_pixels = img_copy.numel()
    num_salt = int(total_pixels * salt_prob)
    num_pepper = int(total_pixels * pepper_prob)

    # Add salt noise
    salt_indices = torch.randint(0, total_pixels, (num_salt,))
    img_copy.view(-1)[salt_indices] = 1.0

    # Add pepper noise
    pepper_indices = torch.randint(0, total_pixels, (num_pepper,))
    img_copy.view(-1)[pepper_indices] = 0.0

    return img_copy

def add_poisson_noise(img):
    img_copy = img.clone()
    noisy_img = torch.poisson(img_copy * 255) / 255
    return noisy_img

def add_speckle_noise(img, noise_factor=0.5):
    img_copy = img.clone()
    noise = torch.randn_like(img_copy) * noise_factor
    noisy_img = img_copy + img_copy * noise
    return torch.clamp(noisy_img, 0., 1.)

def add_uniform_noise(img, noise_factor=0.5):
    img_copy = img.clone()
    noise = torch.rand_like(img_copy) * noise_factor
    noisy_img = img_copy + noise
    return torch.clamp(noisy_img, 0., 1.)

def add_exponential_noise(img, scale=0.1):
    img_copy = img.clone()
    noise = torch.from_numpy(np.random.exponential(scale, img_copy.shape)).float()
    noisy_img = img_copy + noise
    return torch.clamp(noisy_img, 0., 1.)

def add_rayleigh_noise(img, scale=0.1):
    img_copy = img.clone()
    noise = torch.from_numpy(np.random.rayleigh(scale, img_copy.shape)).float()
    noisy_img = img_copy + noise
    return torch.clamp(noisy_img, 0., 1.)

def add_erlang_noise(img, shape=2.0, scale=0.1):
    img_copy = img.clone()
    noise = torch.from_numpy(np.random.gamma(shape, scale, img_copy.shape)).float()
    noisy_img = img_copy + noise
    return torch.clamp(noisy_img, 0., 1.)

def add_brownian_noise(img, noise_factor=0.5):
    img_copy = img.clone()
    brownian_noise = np.zeros_like(img_copy.numpy())
    
    for channel in range(img_copy.shape[0]):
        brownian_noise[channel] = np.cumsum(np.cumsum(np.random.randn(*img_copy.shape[1:]), axis=0), axis=1)
    
    # Normalize and adjust by noise factor
    brownian_noise = brownian_noise - brownian_noise.min()
    brownian_noise = (brownian_noise / brownian_noise.max()) * noise_factor
    
    noisy_img = img_copy + torch.tensor(brownian_noise).float()
    return torch.clamp(noisy_img, 0., 1.)

def add_quantization_noise(img, levels=2):
    img_copy = img.clone()
    step = 1.0 / levels
    quantized = torch.round(img_copy / step) * step
    noise = quantized - img_copy
    noisy_img = img_copy + noise  # Here, instead of just adding quantized, we add the noise.
    return torch.clamp(noisy_img, 0., 1.)

def add_stripe_noise(img, vertical=True, noise_factor=0.5):
    img_copy = img.clone()
    if vertical:
        noise = torch.randn(img_copy.shape[0], 1, img_copy.shape[2]) * noise_factor
        noise = noise.expand_as(img_copy)
    else:  # horizontal
        noise = torch.randn(1, img_copy.shape[1], img_copy.shape[2]) * noise_factor
        noise = noise.expand_as(img_copy)
    noisy_img = img_copy + noise
    return torch.clamp(noisy_img, 0., 1.)

def add_multiplicative_noise(img, noise_factor=0.5):
    img_copy = img.clone()
    noise = 1 + torch.randn_like(img_copy) * noise_factor
    noisy_img = img_copy * noise
    return torch.clamp(noisy_img, 0., 1.)

NOISE_FUNCTIONS = {
    'gaussian': add_gaussian_noise,
    'salt_and_pepper': add_salt_and_pepper_noise,
    'speckle': add_speckle_noise,
    'poisson': add_poisson_noise,
    'uniform': add_uniform_noise,
    'exponential': add_exponential_noise,
    'rayleigh': add_rayleigh_noise,
    'erlang': add_erlang_noise,
    'brownian': add_brownian_noise,
    'quantization': add_quantization_noise,
    'stripe': add_stripe_noise,
    'multiplicative': add_multiplicative_noise
}

def add_selected_noise(img, noise_type='gaussian', **kwargs):
    if noise_type in NOISE_FUNCTIONS:
        return NOISE_FUNCTIONS[noise_type](img, **kwargs)
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")
