import math

import numpy as np
import torch
import torchvision.utils
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage.color import rgb2gray
from skimage.metrics import structural_similarity
from torchvision import transforms
from numpy.random import random
from PIL import Image


def plot_loss(loss):
    plt.figure()
    plt.plot(loss)
    plt.show()
    plt.close('all')


def imgshow(im, cmap=None, rgb_axis=None, dpi=100, figsize=(6.4, 4.8)):
    if isinstance(im, torch.Tensor):
        im = im.to('cpu').detach().cpu().numpy()
    if rgb_axis is not None:
        im = np.moveaxis(im, rgb_axis, -1)
        im = rgb2gray(im)

    plt.figure(dpi=dpi, figsize=figsize)
    norm_obj = Normalize(vmin=im.min(), vmax=im.max())
    plt.imshow(im, norm=norm_obj, cmap=cmap)
    plt.colorbar()
    plt.show()
    plt.close('all')


def imsshow(imgs, titles=None, num_col=5, dpi=100, cmap=None, is_colorbar=False, is_ticks=False):
    '''
    assume imgs's shape is (Nslice, Nx, Ny)
    '''
    num_imgs = len(imgs)
    num_row = math.ceil(num_imgs / num_col)
    fig_width = num_col * 3
    if is_colorbar:
        fig_width += num_col * 1.5
    fig_height = num_row * 3
    fig = plt.figure(dpi=dpi, figsize=(fig_width, fig_height))
    for i in range(num_imgs):
        ax = plt.subplot(num_row, num_col, i + 1)
        im = ax.imshow(imgs[i], cmap=cmap)
        if titles:
            plt.title(titles[i])
        if is_colorbar:
            cax = fig.add_axes([ax.get_position().x1 + 0.01, ax.get_position().y0, 0.01, ax.get_position().height])
            plt.colorbar(im, cax=cax)
        if not is_ticks:
            ax.set_xticks([])
            ax.set_yticks([])
    plt.show()
    plt.close('all')




def make_grid_and_show(ims, nrow=5, cmap=None):
    if isinstance(ims, np.ndarray):
        ims = torch.from_numpy(ims)

    B, C, H, W = ims.shape
    grid_im = torchvision.utils.make_grid(ims, nrow=nrow)
    fig_h, fig_w = nrow * 2 + 1, (B / nrow) + 1
    imgshow(grid_im, cmap=cmap, rgb_axis=0, dpi=200, figsize=(fig_h, fig_w))


def int2preetyStr(num: int):
    s = str(num)
    remain_len = len(s)
    while remain_len - 3 > 0:
        s = s[:remain_len - 3] + ',' + s[remain_len - 3:]
        remain_len -= 3
    return s


def compute_num_params(module, is_trace=False):
    print(int2preetyStr(sum([p.numel() for p in module.parameters()])))
    if is_trace:
        for item in [f"[{int2preetyStr(info[1].numel())}] {info[0]}:{tuple(info[1].shape)}"
                     for info in module.named_parameters()]:
            print(item)


def tonp(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu()
    else:
        return x


def pseudo2real(x):
    """
    :param x: [..., C=2, H, W]
    :return: [..., H, W]
    """
    return (x[..., 0, :, :] ** 2 + x[..., 1, :, :] ** 2) ** 0.5


def complex2pseudo(x):
    """
    :param x: [..., H, W] Complex
    :return: [...., C=2, H, W]
    """
    if isinstance(x, np.ndarray):
        return np.stack([x.real, x.imag], axis=-3)
    elif isinstance(x, torch.Tensor):
        return torch.stack([x.real, x.imag], dim=-3)
    else:
        raise RuntimeError("Unsupported type.")


def pseudo2complex(x):
    """
    :param x:  [..., C=2, H, W]
    :return: [..., H, W] Complex
    """
    if x.size(-3) == 2:
        return x[..., 0, :, :] + x[..., 1, :, :] * 1j
    elif x.size(-3) == 1:  
        return x[..., 0, :, :]
    else:
        raise ValueError("error")


# ================================
# Preprocessing
# ================================
def minmax_normalize(x, eps=1e-8):
    min = x.min()
    max = x.max()
    return (x - min) / (max - min + eps)


# ================================
# kspace and image domain transform
# reference: [ismrmrd-python-tools/transform.py at master · ismrmrd/ismrmrd-python-tools · GitHub](https://github.com/ismrmrd/ismrmrd-python-tools/blob/master/ismrmrdtools/transform.py)
# ================================
def image2kspace(x):
    if isinstance(x, np.ndarray):
        x = np.fft.ifftshift(x, axes=(-2, -1))
        x = np.fft.fft2(x)
        x = np.fft.fftshift(x, axes=(-2, -1))
        return x
    elif isinstance(x, torch.Tensor):
        x = torch.fft.ifftshift(x, dim=(-2, -1))
        x = torch.fft.fft2(x)
        x = torch.fft.fftshift(x, dim=(-2, -1))
        return x
    else:
        raise RuntimeError("Unsupported type.")


def kspace2image(x):
    if isinstance(x, np.ndarray):
        x = np.fft.ifftshift(x, axes=(-2, -1))
        x = np.fft.ifft2(x)
        x = np.fft.fftshift(x, axes=(-2, -1))
        return x
    elif isinstance(x, torch.Tensor):
        x = torch.fft.ifftshift(x, dim=(-2, -1))
        x = torch.fft.ifft2(x)
        x = torch.fft.fftshift(x, dim=(-2, -1))
        return x
    else:
        raise RuntimeError("Unsupported type.")


# ======================================
# Metrics
# ======================================
def compute_mse(x, y):
    """
    REQUIREMENT: `x` and `y` can be any shape, but their shape have to be same
    """
    assert x.dtype == y.dtype and x.shape == y.shape, \
        'x and y is not compatible to compute MSE metric'

    if isinstance(x, np.ndarray):
        mse = np.mean(np.abs(x - y) ** 2)

    elif isinstance(x, torch.Tensor):
        mse = torch.mean(torch.abs(x - y) ** 2)

    else:
        raise RuntimeError(
            'Unsupported object type'
        )
    return mse


def compute_psnr(reconstructed_im, target_im, peak='normalized', is_minmax=False):
    '''
    Image must be of either Integer [0, 255] or Float value [0,1]
    :param peak: 'max' or 'normalize', max_intensity will be the maximum value of target_im if peek == 'max.
          when peek is 'normalized', max_intensity will be the maximum value depend on data representation (in this
          case, we assume your input should be normalized to [0,1])
    REQUIREMENT: `x` and `y` can be any shape, but their shape have to be same
    '''
    assert target_im.dtype == reconstructed_im.dtype and target_im.shape == reconstructed_im.shape, \
        'target_im and reconstructed_im is not compatible to compute PSNR metric'
    assert peak in {'max', 'normalized'}, \
        'peak mode is not supported'

    eps = 1e-8  # to avoid math error in log(x) when x=0

    if is_minmax:
        reconstructed_im = minmax_normalize(reconstructed_im, eps)
        target_im = minmax_normalize(target_im, eps)

    if isinstance(target_im, np.ndarray):
        max_intensity = 255 if target_im.dtype == np.uint8 else 1.0
        max_intensity = np.max(target_im).item() if peak == 'max' else max_intensity
        psnr = 20 * math.log10(max_intensity) - 10 * np.log10(compute_mse(reconstructed_im, target_im) + eps)

    elif isinstance(target_im, torch.Tensor):
        max_intensity = 255 if target_im.dtype == torch.uint8 else 1.0
        max_intensity = torch.max(target_im).item() if peak == 'max' else max_intensity
        psnr = 20 * math.log10(max_intensity) - 10 * torch.log10(compute_mse(reconstructed_im, target_im) + eps)

    else:
        raise RuntimeError(
            'Unsupported object type'
        )
    return psnr


# def compute_ssim(reconstructed_im, target_im, is_minmax=False):
#     """
#     Compute structural similarity index between two batches using skimage library,
#     which only accept 2D-image input. We have to specify where is image's axes.

#     WARNING: this method using skimage's implementation, DOES NOT SUPPORT GRADIENT
#     """
#     assert target_im.dtype == reconstructed_im.dtype and target_im.shape == reconstructed_im.shape, \
#         'target_im and reconstructed_im is not compatible to compute SSIM metric'

#     if isinstance(target_im, np.ndarray):
#         pass
#     elif isinstance(target_im, torch.Tensor):
#         target_im = target_im.detach().to('cpu').numpy()
#         reconstructed_im = reconstructed_im.detach().to('cpu').numpy()
#     else:
#         raise RuntimeError(
#             'Unsupported object type'
#         )
    
#     eps = 1e-8  # to avoid math error in log(x) when x=0

#     if is_minmax:
#         reconstructed_im = minmax_normalize(reconstructed_im, eps)
#         target_im = minmax_normalize(target_im, eps)
    
#     ssim_value = structural_similarity(target_im, reconstructed_im, \
#         gaussian_weights=True, sigma=1.5, use_sample_covariance=False,\
#             data_range= target_im.max() - target_im.min())

#     return ssim_value 

def compute_ssim(reconstructed_im, target_im, is_minmax=False, win_size=7):
    """
    Compute structural similarity index between two batches using skimage library,
    which only accept 2D-image input. We have to specify where is image's axes.

    WARNING: this method using skimage's implementation, DOES NOT SUPPORT GRADIENT
    """
    assert target_im.dtype == reconstructed_im.dtype and target_im.shape == reconstructed_im.shape, \
        'target_im and reconstructed_im is not compatible to compute SSIM metric'

    if isinstance(target_im, np.ndarray):
        pass
    elif isinstance(target_im, torch.Tensor):
        target_im = target_im.detach().to('cpu').numpy()
        reconstructed_im = reconstructed_im.detach().to('cpu').numpy()
    else:
        raise RuntimeError(
            'Unsupported object type'
        )
    
    eps = 1e-8  # to avoid math error in log(x) when x=0

    if is_minmax:
        reconstructed_im = minmax_normalize(reconstructed_im, eps)
        target_im = minmax_normalize(target_im, eps)
    
    ssim_value = structural_similarity(target_im, reconstructed_im, \
        win_size=win_size, gaussian_weights=True, sigma=1.5, use_sample_covariance=False,\
            data_range= target_im.max() - target_im.min())

    return ssim_value


# Convert a PIL Image or numpy.ndarray to tensor.
def ToTensor():
    return torchvision.transforms.ToTensor() 

# Convert image to grayscale.
def Grayscale(num_output_channels=1):
    return torchvision.transforms.Grayscale(num_output_channels=num_output_channels) 

# Normalize a tensor image with mean and standard deviation.
def Normalize(mean, std, inplace=False):
    return torchvision.transforms.Normalize(mean, std, inplace=inplace) 

# Composes several transforms together.
def Compose(transforms):
    return torchvision.transforms.Compose(transforms) 

# Crop the image at the center.
def CenterCrop(size):
    return torchvision.transforms.CenterCrop(size) 

# Pad the given PIL Image on all sides with the given “pad” value.
def Pad(padding, fill=0, padding_mode='constant'):
    return torchvision.transforms.Pad(padding, fill=fill, padding_mode=padding_mode) 

# Crop the given PIL Image at a random location.
def RandomCrop(size, padding=None, pad_if_needed=False, fill=0, padding_mode='constant'):
    return torchvision.transforms.RandomCrop(size, padding=padding, pad_if_needed=pad_if_needed, fill=fill, padding_mode=padding_mode) 

#  Resize the input PIL Image to the given size.
def Resize(size, interpolation=2):
    return torchvision.transforms.Resize(size, interpolation=interpolation)

# Rotate the image by angle.
def RandomRotation(degrees, expand=False, center=None, fill=None):
    return torchvision.transforms.RandomRotation(degrees, expand=expand, center=center, fill=fill)


# Randomly change the brightness, contrast and saturation of an image.
def RandomHorizontalFlip(p=0.5):
    return torchvision.transforms.RandomHorizontalFlip(p=p)


# Randomly change the brightness, contrast and saturation of an image.
def ColorJitter(brightness=0, contrast=0, saturation=0, hue=0):
    return torchvision.transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)

# Randomly change the brightness, contrast and saturation of an image.
def GaussianBlur(kernel_size, sigma=(0.1, 2.0)):
    return torchvision.transforms.GaussianBlur(kernel_size, sigma=sigma)

# Randomly change the brightness, contrast and saturation of an image.
def RandomAffine(degrees, translate=None, scale=None, shear=None):
    return torchvision.transforms.RandomAffine(degrees, translate=translate, scale=scale, shear=shear)

#lr_scheduler

def LambdaLR(optimizer, lr_lambda, last_epoch=-1):
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)

def StepLR(optimizer, step_size, gamma=0.1, last_epoch=-1):
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma=gamma, last_epoch=last_epoch)

def MultiStepLR(optimizer, milestones, gamma=0.1, last_epoch=-1):
    return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=gamma, last_epoch=last_epoch)

def ExponentialLR(optimizer, gamma, last_epoch=-1):
    return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma, last_epoch=last_epoch)

def CosineAnnealingLR(optimizer, T_max, eta_min=0, last_epoch=-1):
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max, eta_min=eta_min, last_epoch=last_epoch)



# def apply_transforms(x):
#     transform_pipeline = transforms.Compose([
#         transforms.RandomHorizontalFlip(p=0.5),
#         transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
#         transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.5),
#         transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5)
#     ])
#     x = [transforms.ToTensor()(img) for img in x]
#     transformed_x = torch.stack([transform_pipeline(img) for img in x])
#     return transformed_x

# def apply_transforms(x):
#     transform_pipeline = transforms.Compose([
#         transforms.RandomHorizontalFlip(p=0.5),
#         transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
#         transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.5),
#         transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5)
#     ])
#     transformed_x = torch.stack([transform_pipeline(transforms.ToPILImage()(img)) for img in x])
#     return transformed_x

def apply_transforms(x):
    batch_size, num_samples, channels, height, width = x.size()
    x_flat = x.view(-1, channels, height, width)
    transformed_samples = []
    for sample in x_flat:
        sample_pil = transforms.ToPILImage()(sample)
        sample_transformed = transforms.Compose([
            # transforms.Grayscale(num_output_channels=1),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.5),
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5),
            transforms.ToTensor()
        ])(sample_pil)
        transformed_samples.append(sample_transformed)
    transformed_x = torch.stack(transformed_samples).view(batch_size, num_samples, channels, height, width)
    return transformed_x


def apply_transforms_labels(x):
    # Add an extra dimension for the channel
    x = x.unsqueeze(2)
    batch_size, num_samples, _, height, width = x.size()
    x_flat = x.view(-1, _, height, width)
    transformed_samples = []
    for sample in x_flat:
        sample_pil = transforms.ToPILImage()(sample)
        sample_transformed = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.5),
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5),
            transforms.ToTensor()
        ])(sample_pil)
        transformed_samples.append(sample_transformed)
    transformed_x = torch.stack(transformed_samples).view(batch_size, num_samples, height, width)
    return transformed_x






# def apply_transforms(x):
#     transform_pipeline = transforms.Compose([
#         transforms.RandomHorizontalFlip(p=0.5),
#         transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
#         transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.5),
#         transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5),
#         transforms.ToTensor()
#     ])
#     x_np = x.numpy() 
#     x_np = np.transpose(x_np, (0, 2, 3, 1))  
#     x_pil = [Image.fromarray((img * 255).astype(np.uint8)) for img in x_np] 
#     transformed_x = torch.stack([transform_pipeline(img) for img in x_pil])
#     return transformed_x
# def apply_transforms(x):
#     transform_pipeline = transforms.Compose([
#         transforms.RandomHorizontalFlip(p=0.5),
#         transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
#         transforms.ToTensor()
#     ])
#     x_np = np.transpose(x, (0, 2, 3, 1))  
#     x_pil = [Image.fromarray((img * 255).astype(np.uint8)) for img in x_np] 
#     x_transformed = [transform_pipeline(img) for img in x_pil]
#     x_tensor = torch.stack(x_transformed)
#     return x_tensor

def lr_scheduler(lr,warmup_epochs, warmup_lr, initial_lr, num_epochs):
    if lr < warmup_epochs:
        return warmup_lr + lr * (initial_lr - warmup_lr) / warmup_epochs
    else:
        return initial_lr * 0.5 * (1 + np.cos((lr - warmup_epochs) * np.pi / (num_epochs - warmup_epochs)))


