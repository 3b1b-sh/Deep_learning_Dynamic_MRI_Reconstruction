"""
BME1301
DO NOT MODIFY anything in this file.
"""
import itertools
import statistics
from typing import Callable

import numpy as np

# from tqdm import tqdm
from matplotlib import pyplot as plt
from tqdm.autonotebook import tqdm  # may raise warning about Jupyter
from tqdm.auto import tqdm  # who needs warnings

import torch
from torch import nn
from torch.utils import data as Data

from .dataset import build_loader
from .utils import imsshow, pseudo2complex, kspace2image, complex2pseudo, pseudo2real, compute_psnr, compute_ssim


class Solver(object):
    def __init__(self,
                 model: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 criterion: Callable,
                 recorder: dict = None,
                 device=None):
        device = device if device is not None else \
            ('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.recorder = recorder
        
        self.model = self.to_device(model)
        self.optimizer = optimizer
        self.criterion = criterion
        

    def _step(self,
              batch: tuple) -> dict:
        raise NotImplementedError()

    def to_device(self, x):
        if isinstance(x, torch.Tensor):
            return x.to(self.device)
        elif isinstance(x, np.ndarray):
            return torch.tensor(x, device=self.device)
        elif isinstance(x, nn.Module):
            return x.to(self.device)
        else:
            raise RuntimeError("Data cannot transfer to correct device.")

    def to_numpy(self, x):
        if isinstance(x, np.ndarray):
            return x
        elif isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        else:
            raise RuntimeError(f"Cannot convert type {type(x)} into numpy array.")

    def train(self,
              epochs: int,
              data_loader,
              *,
              val_loader=None,
              is_plot=True) -> dict:
        val_loss_epochs = []
        train_loss_epochs = []
        pbar_train = tqdm(total=len(data_loader.sampler), unit='img')
        if val_loader is not None:
            pbar_val = tqdm(total=len(val_loader.sampler), desc=f'[Validation] waiting', unit='img')
        for epoch in range(epochs):
            pbar_train.reset()
            pbar_train.set_description(desc=f'[Train] Epoch {epoch + 1}/{epochs}')
            epoch_loss_acc = 0
            epoch_size = 0
            for batch in data_loader:
                self.model.train()
                # clear the gradient
                self.optimizer.zero_grad()
                
                # forward
                step_dict = self._step(batch)
                batch_size = step_dict['batch_size']
                loss = step_dict['loss']

                # backward
                loss.backward()

                # optimize
                self.optimizer.step()

                # update information
                loss_value = loss.item()
                epoch_loss_acc += loss_value
                epoch_size += batch_size
                pbar_train.update(batch_size)
                pbar_train.set_postfix(loss=loss_value / batch_size)

            epoch_avg_loss = epoch_loss_acc / epoch_size
            pbar_train.set_postfix(epoch_avg_loss=epoch_avg_loss)
            train_loss_epochs.append(epoch_avg_loss)

            # validate if `val_loader` is specified
            if val_loader is not None:
                pbar_val.reset()
                pbar_val.set_description(desc=f'[Validation] Epoch {epoch + 1}/{epochs}')
                val_avg_loss = self.validate(val_loader, pbar=pbar_val, is_compute_metrics=False)
                val_loss_epochs.append(val_avg_loss)

        pbar_train.close()
        if val_loader is not None:
            pbar_val.close()
        train_loss_epochs = torch.tensor(train_loss_epochs).numpy()
        val_loss_epochs = torch.tensor(val_loss_epochs).numpy()

        plt.figure()
        plt.plot(list(range(1, epochs + 1)), train_loss_epochs, label='train')
        if val_loader is not None:
            plt.plot(list(range(1, epochs + 1)), val_loss_epochs, label='validation')
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.show()
        plt.close('all')

    def validate(self, data_loader, *, pbar=None, is_compute_metrics=True) -> float:
        """
        :param pbar: when pbar is specified, do not print average loss
        :return:
        """
        metrics_acc = {}
        loss_acc = 0
        size_acc = 0
        is_need_log = (pbar is None)
        if pbar is None:
            pbar = tqdm(total=len(data_loader.sampler), desc=f'[Validation]', unit='img')
        for batch in data_loader:
            self.model.eval()

            # forward
            step_dict = self._step(batch, is_compute_metrics=is_compute_metrics)
            batch_size = step_dict['batch_size']
            loss = step_dict['loss']
            loss_value = loss.item()

            # aggregate metrics
            metrics_acc = self._aggregate_metrics(metrics_acc, step_dict)

            # update information
            loss_acc += loss_value
            size_acc += batch_size
            pbar.update(batch_size)
            pbar.set_postfix(loss=loss_value)

        val_avg_loss = loss_acc / size_acc
        pbar.set_postfix(val_avg_loss=val_avg_loss)
        if is_need_log:
            pbar.close()  # destroy newly created pbar
            print('=' * 30 + ' Measurements' + '=' * 30)
            for k, v in metrics_acc.items():
                print(f"[{k}] {v / size_acc}")
        else:
            return val_avg_loss

    def _aggregate_metrics(self, metrics_acc: dict, step_dict: dict):
        batch_size = step_dict['batch_size']
        for k, v in step_dict.items():
            if k[:7] == 'metric_':
                value = v * batch_size
                metric_name = k[7:]
                if metric_name not in metrics_acc:
                    metrics_acc[metric_name] = value
                else:
                    metrics_acc[metric_name] += value
        return metrics_acc

    def visualize(self, data_loader, idx, net):
        raise NotImplementedError()

    def get_recorder(self) -> dict:
        return self.recorder


class Lab1Solver(Solver):
    def _step(self, batch, is_compute_metrics=True) -> dict:
        k_und, und_mask, image_gt = batch

        k_und = self.to_device(k_und)  # [B, C=2, H, W]
        und_mask = self.to_device(und_mask)  # [B, H, W]
        image_gt = self.to_device(image_gt)  # [B, C=2, H, W]
        B, C, H, W = k_und.shape

        im_recon = self.model(k_und, und_mask)  # [B, C=2, H, W]
        loss = self.criterion(im_recon, image_gt)

        step_dict = {
            'loss': loss,
            'batch_size': B
        }

        # ============ compute metrics
        if not self.model.training and is_compute_metrics:
            im_recon = pseudo2real(im_recon)  # [B, H, W]
            image_gt = pseudo2real(image_gt)  # [B, H, W]
            psnr_images = [compute_psnr(im_recon[i], image_gt[i], is_minmax=True).item() for i in range(B)]
            ssim_images = [compute_ssim(im_recon[i], image_gt[i], is_minmax=True).item() for i in range(B)]
            step_dict['metric_avg_PSNR'] = statistics.mean(psnr_images)
            step_dict['metric_avg_SSIM'] = statistics.mean(ssim_images)

        return step_dict

    def visualize(self, data_loader, idx, *, dpi=100):
        if idx < 0 or idx > len(data_loader) * data_loader.batch_size:
            raise RuntimeError("idx is out of range.")

        batch_idx = idx // data_loader.batch_size
        batch_offset = idx - batch_idx * data_loader.batch_size

        batch = next(itertools.islice(data_loader, batch_idx, None))
        k_und, und_mask, image_gt = batch

        k_und = self.to_device(k_und)  # [B, C=2, H, W]
        und_mask = self.to_device(und_mask)  # [B, H, W]
        image_gt = self.to_device(image_gt)  # [B, C=2, H, W]
        B, C, H, W = k_und.shape

        self.model.eval()
        im_recon = self.model(k_und, und_mask)  # [B, C=2, H, W]

        im_und = torch.abs(kspace2image(pseudo2complex(k_und)))  # [B, H, W]
        image_gt = pseudo2real(image_gt)  # [B, H, W]
        im_recon = pseudo2real(im_recon)  # [B, H, W]

        im_und = self.to_numpy(im_und[batch_offset])
        image_gt = self.to_numpy(image_gt[batch_offset])
        im_recon = self.to_numpy(im_recon[batch_offset])

        imsshow([image_gt, im_und, im_recon],
                titles=['Fully sampled',
                        f"Under sampled (PSNR {compute_psnr(im_und, image_gt, is_minmax=True):.2f})",
                        f"Reconstruction (PSNR {compute_psnr(im_recon, image_gt, is_minmax=True):.2f})"],
                num_col=3,
                dpi=dpi,
                is_colorbar=True,
                cmap='gray')
