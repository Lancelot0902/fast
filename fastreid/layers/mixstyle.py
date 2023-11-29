import random
from contextlib import contextmanager
import torch
import torch.nn as nn
import random
from math import sqrt
import torch.fft

import numpy as np


def deactivate_mixstyle(m):
    if type(m) == MixStyle:
        m.set_activation_status(False)


def activate_mixstyle(m):
    if type(m) == MixStyle:
        m.set_activation_status(True)


def random_mixstyle(m):
    if type(m) == MixStyle:
        m.update_mix_method('random')


def crossdomain_mixstyle(m):
    if type(m) == MixStyle:
        m.update_mix_method('crossdomain')


@contextmanager
def run_without_mixstyle(model):
    # Assume MixStyle was initially activated
    try:
        model.apply(deactivate_mixstyle)
        yield
    finally:
        model.apply(activate_mixstyle)


@contextmanager
def run_with_mixstyle(model, mix=None):
    # Assume MixStyle was initially deactivated
    if mix == 'random':
        model.apply(random_mixstyle)

    elif mix == 'crossdomain':
        model.apply(crossdomain_mixstyle)

    try:
        model.apply(activate_mixstyle)
        yield
    finally:
        model.apply(deactivate_mixstyle)


class MixStyle(nn.Module):
    """MixStyle.

    Reference:
      Zhou et al. Domain Generalization with MixStyle. ICLR 2021.
    """

    def __init__(self, p=0.5, ratio=1.0, eps=1e-6, factor=1.0): # p = 0.5
        """
        Args:
          p (float): probability of using MixStyle.
          alpha (float): parameter of the Beta distribution.
          eps (float): scaling parameter to avoid numerical issues.
          mix (str): how to mix.
        """
        super().__init__()
        self.p = p
        self.ratio = ratio
        self.eps = eps
        self.factor = factor
        
    def forward(self, x):
        """
        if not self.training or not self._activated:
            return x

        if random.random() > self.p:
            return x
        """

        B, C, H, W = x.shape
        
        x = torch.fft.rfft2(x, dim=(2,3), norm='ortho')
        
        if random.random() > self.p:
            return x
        
        x_abs, x_pha = torch.abs(x), torch.angle(x)
        
        x_abs = torch.fft.fftshift(x_abs, dim=(2))
        
        h_crop = int(H * sqrt(self.ratio))
        w_crop = int(W * sqrt(self.ratio))
        h_start = H // 2 - h_crop // 2
        w_start = 0 
        
        x_abs_ = x_abs.clone()
        
        miu_of_elem = torch.mean(x_abs_[:, :, h_start:h_start + h_crop, w_start:w_start + w_crop], dim=0, 
                                 keepdim=True)
        var_of_elem = torch.var(x_abs_[:, :, h_start:h_start + h_crop, w_start:w_start + w_crop, ], dim=0,
                                keepdim=True)
        sig_of_elem = (var_of_elem + self.eps).sqrt()   # 1 x C x H x W
        
        epsilon_sig = torch.randn_like(x_abs[:, :, h_start:h_start + h_crop, w_start:w_start + w_crop])  # BxHxWxC N(0,1)
        gamma = epsilon_sig * sig_of_elem * self.factor
        
        x_abs[:, :, h_start:h_start + h_crop, w_start:w_start + w_crop] = \
                        x_abs[:, :, h_start:h_start + h_crop, w_start:w_start + w_crop] + gamma
                        
        x_abs = torch.fft.ifftshift(x_abs, dim=(2))
        x_mix = x_abs * (np.e ** (1j * x_pha))
        
        x = x_mix
        x = torch.fft.irfft2(x, s=(H,W), dim=(2,3), norm='ortho')
        return x
