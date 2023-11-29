# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import logging

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

__all__ = ["IBN", "get_norm"]


class BatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, weight_freeze=False, bias_freeze=False, weight_init=1.0,
                 bias_init=0.0, **kwargs):
        super().__init__(num_features, eps=eps, momentum=momentum)
        if weight_init is not None: nn.init.constant_(self.weight, weight_init)
        if bias_init is not None: nn.init.constant_(self.bias, bias_init)
        self.weight.requires_grad_(not weight_freeze)
        self.bias.requires_grad_(not bias_freeze)


class SyncBatchNorm(nn.SyncBatchNorm):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, weight_freeze=False, bias_freeze=False, weight_init=1.0,
                 bias_init=0.0):
        super().__init__(num_features, eps=eps, momentum=momentum)
        if weight_init is not None: nn.init.constant_(self.weight, weight_init)
        if bias_init is not None: nn.init.constant_(self.bias, bias_init)
        self.weight.requires_grad_(not weight_freeze)
        self.bias.requires_grad_(not bias_freeze)


class IBN(nn.Module):
    def __init__(self, planes, bn_norm, **kwargs):
        super(IBN, self).__init__()
        half1 = int(planes / 2)
        self.half = half1
        half2 = planes - half1
        self.IN = nn.InstanceNorm2d(half1, affine=True)
        self.BN = get_norm(bn_norm, half2, **kwargs)

    def forward(self, x):
        split = torch.split(x, self.half, 1)
        out1 = self.IN(split[0].contiguous())
        out2 = self.BN(split[1].contiguous())
        out = torch.cat((out1, out2), 1)
        return out


class GhostBatchNorm(BatchNorm):
    def __init__(self, num_features, num_splits=1, **kwargs):
        super().__init__(num_features, **kwargs)
        self.num_splits = num_splits
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, input):
        N, C, H, W = input.shape
        if self.training or not self.track_running_stats:
            self.running_mean = self.running_mean.repeat(self.num_splits)
            self.running_var = self.running_var.repeat(self.num_splits)
            outputs = F.batch_norm(
                input.view(-1, C * self.num_splits, H, W), self.running_mean, self.running_var,
                self.weight.repeat(self.num_splits), self.bias.repeat(self.num_splits),
                True, self.momentum, self.eps).view(N, C, H, W)
            self.running_mean = torch.mean(self.running_mean.view(self.num_splits, self.num_features), dim=0)
            self.running_var = torch.mean(self.running_var.view(self.num_splits, self.num_features), dim=0)
            return outputs
        else:
            return F.batch_norm(
                input, self.running_mean, self.running_var,
                self.weight, self.bias, False, self.momentum, self.eps)


class FrozenBatchNorm(nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.
    It contains non-trainable buffers called
    "weight" and "bias", "running_mean", "running_var",
    initialized to perform identity transformation.
    The pre-trained backbone models from Caffe2 only contain "weight" and "bias",
    which are computed from the original four parameters of BN.
    The affine transform `x * weight + bias` will perform the equivalent
    computation of `(x - running_mean) / sqrt(running_var) * weight + bias`.
    When loading a backbone model from Caffe2, "running_mean" and "running_var"
    will be left unchanged as identity transformation.
    Other pre-trained backbone models may contain all 4 parameters.
    The forward is implemented by `F.batch_norm(..., training=False)`.
    """

    _version = 3

    def __init__(self, num_features, eps=1e-5, **kwargs):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.register_buffer("weight", torch.ones(num_features))
        self.register_buffer("bias", torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features) - eps)

    def forward(self, x):
        if x.requires_grad:
            # When gradients are needed, F.batch_norm will use extra memory
            # because its backward op computes gradients for weight/bias as well.
            scale = self.weight * (self.running_var + self.eps).rsqrt()
            bias = self.bias - self.running_mean * scale
            scale = scale.reshape(1, -1, 1, 1)
            bias = bias.reshape(1, -1, 1, 1)
            return x * scale + bias
        else:
            # When gradients are not needed, F.batch_norm is a single fused op
            # and provide more optimization opportunities.
            return F.batch_norm(
                x,
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                training=False,
                eps=self.eps,
            )

    def _load_from_state_dict(
            self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        version = local_metadata.get("version", None)

        if version is None or version < 2:
            # No running_mean/var in early versions
            # This will silent the warnings
            if prefix + "running_mean" not in state_dict:
                state_dict[prefix + "running_mean"] = torch.zeros_like(self.running_mean)
            if prefix + "running_var" not in state_dict:
                state_dict[prefix + "running_var"] = torch.ones_like(self.running_var)

        if version is not None and version < 3:
            logger = logging.getLogger(__name__)
            logger.info("FrozenBatchNorm {} is upgraded to version 3.".format(prefix.rstrip(".")))
            # In version < 3, running_var are used without +eps.
            state_dict[prefix + "running_var"] -= self.eps

        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def __repr__(self):
        return "FrozenBatchNorm2d(num_features={}, eps={})".format(self.num_features, self.eps)

    @classmethod
    def convert_frozen_batchnorm(cls, module):
        """
        Convert BatchNorm/SyncBatchNorm in module into FrozenBatchNorm.
        Args:
            module (torch.nn.Module):
        Returns:
            If module is BatchNorm/SyncBatchNorm, returns a new module.
            Otherwise, in-place convert module and return it.
        Similar to convert_sync_batchnorm in
        https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/batchnorm.py
        """
        bn_module = nn.modules.batchnorm
        bn_module = (bn_module.BatchNorm2d, bn_module.SyncBatchNorm)
        res = module
        if isinstance(module, bn_module):
            res = cls(module.num_features)
            if module.affine:
                res.weight.data = module.weight.data.clone().detach()
                res.bias.data = module.bias.data.clone().detach()
            res.running_mean.data = module.running_mean.data
            res.running_var.data = module.running_var.data
            res.eps = module.eps
        else:
            for name, child in module.named_children():
                new_child = cls.convert_frozen_batchnorm(child)
                if new_child is not child:
                    res.add_module(name, new_child)
        return res


class SwitchWhiten2d(Module):
    """Switchable Whitening.

    Args:
        num_features (int): Number of channels.
        num_pergroup (int): Number of channels for each whitening group.
        sw_type (int): Switchable whitening type, from {2, 3, 5}.
            sw_type = 2: BW + IW
            sw_type = 3: BW + IW + LN
            sw_type = 5: BW + IW + BN + IN + LN
        T (int): Number of iterations for iterative whitening.
        tie_weight (bool): Use the same importance weight for mean and
            covariance or not.
    """

    def __init__(self,
                 num_features,
                 num_pergroup=16,
                 sw_type=2,
                 T=5,
                 tie_weight=False,
                 eps=1e-5,
                 momentum=0.99,
                 affine=True):
        super(SwitchWhiten2d, self).__init__()
        if sw_type not in [2, 3, 5]:
            raise ValueError('sw_type should be in [2, 3, 5], '
                             'but got {}'.format(sw_type))
        assert num_features % num_pergroup == 0
        self.num_features = num_features    # 64
        self.num_pergroup = num_pergroup    # 16
        self.num_groups = num_features // num_pergroup  # 4
        self.sw_type = sw_type
        self.T = T
        self.tie_weight = tie_weight
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        num_components = sw_type

        self.sw_mean_weight = Parameter(torch.ones(num_components))
        if not self.tie_weight:
            self.sw_var_weight = Parameter(torch.ones(num_components))
        else:
            self.register_parameter('sw_var_weight', None)

        if self.affine:
            self.weight = Parameter(torch.ones(num_features))
            self.bias = Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.register_buffer('running_mean',
                             torch.zeros(self.num_groups, num_pergroup, 1))
        self.register_buffer(
            'running_cov',
            torch.eye(num_pergroup).unsqueeze(0).repeat(self.num_groups, 1, 1))

        self.reset_parameters()

    def reset_parameters(self):
        self.running_mean.zero_()
        self.running_cov.zero_()
        nn.init.ones_(self.sw_mean_weight)
        if not self.tie_weight:
            nn.init.ones_(self.sw_var_weight)
        if self.affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def __repr__(self):
        return ('{name}({num_features}, num_pergroup={num_pergroup}, '
                'sw_type={sw_type}, T={T}, tie_weight={tie_weight}, '
                'eps={eps}, momentum={momentum}, affine={affine})'.format(
                    name=self.__class__.__name__, **self.__dict__))

    def forward(self, x):
        N, C, H, W = x.size()
        c, g = self.num_pergroup, self.num_groups

        in_data_t = x.transpose(0, 1).contiguous()
        # g x c x (N x H x W)
        in_data_t = in_data_t.view(g, c, -1)

        # calculate batch mean and covariance
        if self.training:
            # g x c x 1
            mean_bn = in_data_t.mean(-1, keepdim=True)
            in_data_bn = in_data_t - mean_bn
            # g x c x c
            cov_bn = torch.bmm(in_data_bn,
                               in_data_bn.transpose(1, 2)).div(H * W * N)

            self.running_mean.mul_(self.momentum)
            self.running_mean.add_((1 - self.momentum) * mean_bn.data)
            self.running_cov.mul_(self.momentum)
            self.running_cov.add_((1 - self.momentum) * cov_bn.data)
        else:
            mean_bn = torch.autograd.Variable(self.running_mean)
            cov_bn = torch.autograd.Variable(self.running_cov)

        mean_bn = mean_bn.view(1, g, c, 1).expand(N, g, c, 1).contiguous()
        mean_bn = mean_bn.view(N * g, c, 1)
        cov_bn = cov_bn.view(1, g, c, c).expand(N, g, c, c).contiguous()
        cov_bn = cov_bn.view(N * g, c, c)

        # (N x g) x c x (H x W)
        in_data = x.view(N * g, c, -1)

        eye = in_data.data.new().resize_(c, c)
        eye = torch.nn.init.eye_(eye).view(1, c, c).expand(N * g, c, c)

        # calculate other statistics
        # (N x g) x c x 1
        mean_in = in_data.mean(-1, keepdim=True)
        x_in = in_data - mean_in
        # (N x g) x c x c
        cov_in = torch.bmm(x_in, torch.transpose(x_in, 1, 2)).div(H * W)
        if self.sw_type in [3, 5]:
            x = x.view(N, -1)
            mean_ln = x.mean(-1, keepdim=True).view(N, 1, 1, 1)
            mean_ln = mean_ln.expand(N, g, 1, 1).contiguous().view(N * g, 1, 1)
            var_ln = x.var(-1, keepdim=True).view(N, 1, 1, 1)
            var_ln = var_ln.expand(N, g, 1, 1).contiguous().view(N * g, 1, 1)
            var_ln = var_ln * eye
        if self.sw_type == 5:
            var_bn = torch.diag_embed(torch.diagonal(cov_bn, dim1=-2, dim2=-1))
            var_in = torch.diag_embed(torch.diagonal(cov_in, dim1=-2, dim2=-1))

        # calculate weighted average of mean and covariance
        softmax = nn.Softmax(0)
        mean_weight = softmax(self.sw_mean_weight)
        if not self.tie_weight:
            var_weight = softmax(self.sw_var_weight)
        else:
            var_weight = mean_weight

        # BW + IW
        if self.sw_type == 2:
            # (N x g) x c x 1
            mean = mean_weight[0] * mean_bn + mean_weight[1] * mean_in
            cov = var_weight[0] * cov_bn + var_weight[1] * cov_in + \
                self.eps * eye
        # BW + IW + LN
        elif self.sw_type == 3:
            mean = mean_weight[0] * mean_bn + \
                mean_weight[1] * mean_in + mean_weight[2] * mean_ln
            cov = var_weight[0] * cov_bn + var_weight[1] * cov_in + \
                var_weight[2] * var_ln + self.eps * eye
        # BW + IW + BN + IN + LN
        elif self.sw_type == 5:
            mean = (mean_weight[0] + mean_weight[2]) * mean_bn + \
                (mean_weight[1] + mean_weight[3]) * mean_in + \
                mean_weight[4] * mean_ln
            cov = var_weight[0] * cov_bn + var_weight[1] * cov_in + \
                var_weight[0] * var_bn + var_weight[1] * var_in + \
                var_weight[4] * var_ln + self.eps * eye

        # perform whitening using Newton's iteration
        Ng, c, _ = cov.size()
        P = torch.eye(c).to(cov).expand(Ng, c, c)
        # reciprocal of trace of covariance
        rTr = (cov * P).sum((1, 2), keepdim=True).reciprocal_()
        cov_N = cov * rTr
        for k in range(self.T):
            P = torch.baddbmm(1.5, P, -0.5, torch.matrix_power(P, 3), cov_N)
        # whiten matrix: the matrix inverse of covariance, i.e., cov^{-1/2}
        wm = P.mul_(rTr.sqrt())

        x_hat = torch.bmm(wm, in_data - mean)
        x_hat = x_hat.view(N, C, H, W)
        if self.affine:
            x_hat = x_hat * self.weight.view(1, self.num_features, 1, 1) + \
                self.bias.view(1, self.num_features, 1, 1)

        return x_hat


def get_norm(norm, out_channels, **kwargs):
    """
    Args:
        norm (str or callable): either one of BN, GhostBN, FrozenBN, GN or SyncBN;
            or a callable that takes a channel number and returns
            the normalization layer as a nn.Module
        out_channels: number of channels for normalization layer

    Returns:
        nn.Module or None: the normalization layer
    """
    if isinstance(norm, str):
        if len(norm) == 0:
            return None
        norm = {
            "BN": BatchNorm,
            "syncBN": SyncBatchNorm,
            "GhostBN": GhostBatchNorm,
            "FrozenBN": FrozenBatchNorm,
            "GN": lambda channels, **args: nn.GroupNorm(32, channels),
            "SW": SwitchWhiten2d,
        }[norm]
    return norm(out_channels, **kwargs)
