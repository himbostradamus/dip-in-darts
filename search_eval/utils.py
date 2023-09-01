
import nni
import torch
import nni.retiarii.nn.pytorch as nn

from skimage.metrics import peak_signal_noise_ratio as skimage_psnr

from torchvision import transforms, datasets


def psnr(image_true, image_test):
    # Convert PyTorch tensors to NumPy arrays
    if torch.is_tensor(image_true):
        image_true = image_true.detach().cpu().numpy()
    if torch.is_tensor(image_test):
        image_test = image_test.detach().cpu().numpy()
    return skimage_psnr(image_true, image_test)

def add_noise(img, noise_factor=0.5):
    """Add random noise to an image."""
    noise = torch.randn_like(img) * noise_factor
    noisy_img = img + noise
    return torch.clamp(noisy_img, 0., 1.)

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
                psnr_value = psnr(clean_img, denoised_output)
            print('Iteration: {}\tLoss: {:.6f}\tPSNR: {:.6f} dB'.format(iteration, loss.item(), psnr_value))
            nni.report_intermediate_result(psnr_value)
    return output

def evaluate_denoising(denoised_img, clean_img):
    # We no longer need the model in an eval state or any forward pass here
    # because the denoised image is already generated and passed to the function.

    # consider deleting this function and just using psnr() instead
    return psnr(clean_img, denoised_img)

def main_evaluation(model_cls):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # Instantiate model
    model = model_cls().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    img, _ = dataset[0]  # Original clean image
    noisy_img = add_noise(img).unsqueeze(0).to(device)  # Noisy version of image

    # img_numpy = generate_phantom(resolution=6)
    # img = phantom_to_torch(img_numpy)
    # noisy_img = add_selected_noise(img, noise_type=noise_name)

    # Denoise the image for a set number of iterations
    denoised_img = deep_image_prior_denoising(model, noisy_img, img.unsqueeze(0).to(device), device, optimizer)

    # Evaluate the PSNR of the denoised image
    psnr_value = evaluate_denoising(denoised_img, img.unsqueeze(0).to(device))
    print('PSNR: {:.6f} dB'.format(psnr_value))

    # Report final PSNR to NNI
    nni.report_final_result(psnr_value.item())