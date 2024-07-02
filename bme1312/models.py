import torch
from torch import nn

from bme1312.utils import image2kspace, kspace2image, pseudo2real, pseudo2complex, complex2pseudo
from CS_mask import cartesian_mask


class DataConsistencyLayer(nn.Module):
    """
    This class support different types k-space data consistency
    """

    def __init__(self, is_data_fidelity=False):
        super().__init__()
        self.is_data_fidelity = is_data_fidelity
        if is_data_fidelity:
            self.data_fidelity = nn.Parameter(torch.randn(1))

    def data_consistency(self, k, k0, mask):
        """
        :param k: input k-space (reconstructed kspace, 2D-Fourier transform of im)
        :param k0: initially sampled k-space
        :param mask: sampling pattern
        """
        mask = cartesian_mask(shape=(2, 20, 192, 192), acc=6, sample_n=10, centred=True)
        mask_tensor = torch.from_numpy(mask).to(k.device)  # Convert mask to tensor and ensure it's on the same device as k
        if self.is_data_fidelity:
            v = self.is_data_fidelity
            k_dc = (1 - mask_tensor) * k + mask_tensor * (k + v * k0 / (1 + v))
        else:
            k_dc = (1 - mask_tensor) * k + mask_tensor * k0
        return k_dc

    def forward(self, im, k0, mask):
        """
        im   - Image in pseudo-complex [B, C=2, H, W]
        k0   - original under-sampled Kspace in pseudo-complex [B, C=2, H, W]
        mask - mask for Kspace in Real [B, H, W]
        """
        # mask need to add one axis to broadcast to pseudo-complex channel
        k = image2kspace(pseudo2complex(im))  # [B, H, W] Complex
        k0 = pseudo2complex(k0)
        k_dc = self.data_consistency(k, k0, mask)  # [B, H, W] Complex
        im_dc = complex2pseudo(kspace2image(k_dc))  # [B, C=2, H, W]

        return im_dc


