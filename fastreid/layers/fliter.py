import torch
import torch.nn as nn

import torch.fft
import random

import math
from math import sqrt

import numpy as np


class GlobalFilter(nn.Module):
    def __init__(self, dim, h=14, w=8, out_planes=64, patch_size=16,
                 mask_radio=0.1, mask_alpha=0.5,
                 noise_mode=1,
                 uncertainty_model=0, perturb_prob=0.5,
                 uncertainty_factor=1.0,
                 noise_layer_flag=0, gauss_or_uniform=0, ):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(h, w, dim, 2, dtype=torch.float32) * 0.02)
        self.w = w
        self.h = h

        self.conv = nn.Conv2d(dim, dim, kernel_size=patch_size, stride=patch_size, bias=False)

        self.mask_radio = mask_radio

        self.noise_mode = noise_mode
        self.noise_layer_flag = noise_layer_flag

        self.alpha = mask_alpha

        self.eps = 1e-6
        self.factor = uncertainty_factor
        self.uncertainty_model = uncertainty_model
        self.p = perturb_prob
        self.gauss_or_uniform = gauss_or_uniform

    def _reparameterize(self, mu, std, epsilon_norm):
        # epsilon = torch.randn_like(std) * self.factor
        epsilon = epsilon_norm * self.factor
        mu_t = mu + epsilon * std
        return mu_t

    def spectrum_noise(self, img_fft, ratio=1.0, noise_mode=1,
                       uncertainty_model=0, gauss_or_uniform=0):
        """Input image size: ndarray of [H, W, C]"""
        """noise_mode: 1 amplitude; 2: phase 3:both"""
        """uncertainty_model: 1 batch-wise modeling 2: channel-wise modeling 3:token-wise modeling"""
        if random.random() > self.p:
            return img_fft
        batch_size, h, w, c = img_fft.shape

        img_abs, img_pha = torch.abs(img_fft), torch.angle(img_fft)

        img_abs = torch.fft.fftshift(img_abs, dim=(1))

        h_crop = int(h * sqrt(ratio))
        w_crop = int(w * sqrt(ratio))
        h_start = h // 2 - h_crop // 2
        w_start = 0

        img_abs_ = img_abs.clone()
        if noise_mode != 0:
            if uncertainty_model != 0:
                if uncertainty_model == 1:
                    # batch level modeling
                    miu = torch.mean(img_abs_[:, h_start:h_start + h_crop, w_start:w_start + w_crop, :], dim=(1, 2),
                                     keepdim=True)
                    var = torch.var(img_abs_[:, h_start:h_start + h_crop, w_start:w_start + w_crop, :], dim=(1, 2),
                                    keepdim=True)
                    sig = (var + self.eps).sqrt()  # Bx1x1xC

                    var_of_miu = torch.var(miu, dim=0, keepdim=True)
                    var_of_sig = torch.var(sig, dim=0, keepdim=True)
                    sig_of_miu = (var_of_miu + self.eps).sqrt().repeat(miu.shape[0], 1, 1, 1)
                    sig_of_sig = (var_of_sig + self.eps).sqrt().repeat(miu.shape[0], 1, 1, 1)  # Bx1x1xC

                    if gauss_or_uniform == 0:
                        epsilon_norm_miu = torch.randn_like(sig_of_miu)  # N(0,1)
                        epsilon_norm_sig = torch.randn_like(sig_of_sig)

                        miu_mean = miu
                        sig_mean = sig

                        beta = self._reparameterize(mu=miu_mean, std=sig_of_miu, epsilon_norm=epsilon_norm_miu)
                        gamma = self._reparameterize(mu=sig_mean, std=sig_of_sig, epsilon_norm=epsilon_norm_sig)
                    elif gauss_or_uniform == 1:
                        epsilon_norm_miu = torch.rand_like(sig_of_miu) * 2 - 1.  # U(-1,1)
                        epsilon_norm_sig = torch.rand_like(sig_of_sig) * 2 - 1.
                        beta = self._reparameterize(mu=miu, std=sig_of_miu, epsilon_norm=epsilon_norm_miu)
                        gamma = self._reparameterize(mu=sig, std=sig_of_sig, epsilon_norm=epsilon_norm_sig)
                    else:
                        epsilon_norm_miu = torch.randn_like(sig_of_miu)  # N(0,1)
                        epsilon_norm_sig = torch.randn_like(sig_of_sig)
                        beta = self._reparameterize(mu=miu, std=1., epsilon_norm=epsilon_norm_miu)
                        gamma = self._reparameterize(mu=sig, std=1., epsilon_norm=epsilon_norm_sig)

                    # adjust statistics for each sample
                    img_abs[:, h_start:h_start + h_crop, w_start:w_start + w_crop, :] = gamma * (
                            img_abs[:, h_start:h_start + h_crop, w_start:w_start + w_crop, :] - miu) / sig + beta

                elif uncertainty_model == 2:
                    # element level modeling
                    miu_of_elem = torch.mean(img_abs_[:, h_start:h_start + h_crop, w_start:w_start + w_crop, :], dim=0,
                                             keepdim=True)
                    var_of_elem = torch.var(img_abs_[:, h_start:h_start + h_crop, w_start:w_start + w_crop, :], dim=0,
                                            keepdim=True)
                    sig_of_elem = (var_of_elem + self.eps).sqrt()  # 1xHxWxC

                    if gauss_or_uniform == 0:
                        epsilon_sig = torch.randn_like(
                            img_abs[:, h_start:h_start + h_crop, w_start:w_start + w_crop, :])  # BxHxWxC N(0,1)
                        gamma = epsilon_sig * sig_of_elem * self.factor
                    elif gauss_or_uniform == 1:
                        epsilon_sig = torch.rand_like(
                            img_abs[:, h_start:h_start + h_crop, w_start:w_start + w_crop, :]) * 2 - 1.  # U(-1,1)
                        gamma = epsilon_sig * sig_of_elem * self.factor
                    else:
                        epsilon_sig = torch.randn_like(
                            img_abs[:, h_start:h_start + h_crop, w_start:w_start + w_crop, :])  # BxHxWxC N(0,1)
                        gamma = epsilon_sig * self.factor

                    img_abs[:, h_start:h_start + h_crop, w_start:w_start + w_crop, :] = \
                        img_abs[:, h_start:h_start + h_crop, w_start:w_start + w_crop, :] + gamma
        img_abs = torch.fft.ifftshift(img_abs, dim=(1))  # recover
        img_mix = img_abs * (np.e ** (1j * img_pha))
        return img_mix

    def forward(self, x, spatial_size=None):
        x = self.conv(x).flatten(2).transpose(1,2)  # B C H W -> B H×W C
        B, N, C = x.shape
        if spatial_size is None:
            a = b = int(math.sqrt(N))
        else:
            a, b = spatial_size

        x = x.view(B, a, b, C)
        # x = x.to(torch.float32)
        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')

        if self.training:
            if self.noise_mode != 0 and self.noise_layer_flag == 1:
                x = self.spectrum_noise(x, ratio=self.mask_radio, noise_mode=self.noise_mode,
                                        uncertainty_model=self.uncertainty_model,
                                        gauss_or_uniform=self.gauss_or_uniform)
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
        x = torch.fft.irfft2(x, s=(a, b), dim=(1, 2), norm='ortho')
        x = x.reshape(B, N, C)
        x = x.view(B, self.h, self.w, C).permute(0, 3, 1, 2)
        return x