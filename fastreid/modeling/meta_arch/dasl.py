# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch
from torch import nn
import torch.nn.functional as F

from fastreid.config import configurable
from fastreid.modeling.backbones import build_backbone
from fastreid.modeling.heads import build_heads
from fastreid.modeling.losses import *
from .build import META_ARCH_REGISTRY
from ...utils.history_buffer import MemoryBank
from ...utils.compute_dist import build_dist
from fastreid.layers import pooling, any_softmax
from fastreid.layers.weight_init import weights_init_kaiming


@META_ARCH_REGISTRY.register()
class dasl(nn.Module):
    """
    Baseline architecture. Any models that contains the following two components:
    1. Per-image feature extraction (aka backbone)
    2. Per-image feature aggregation and loss computation
    """

    @configurable
    def __init__(
            self,
            *,
            cfg_,
            backbone,
            heads,
            heads_expert1,
            heads_expert2,
            heads_expert3,
            pixel_mean,
            pixel_std,
            loss_kwargs=None
    ):
        """
        NOTE: this interface is experimental.

        Args:
            backbone:
            heads:
            pixel_mean:
            pixel_std:
        """
        super().__init__()

        self.cfg_ = cfg_

        # backbone
        self.backbone = backbone

        # head
        self.heads = heads
        self.heads_expert1 = heads_expert1
        self.heads_expert2 = heads_expert2
        self.heads_expert3 = heads_expert3
        self.lmda = 0.5

        self.MB = MemoryBank(self.cfg_)

        self.weights1 = nn.Parameter(torch.ones(cfg_.DATASETS.CLASS[0]))
        torch.nn.init.normal_(self.weights1, mean=0, std=1)
        nn.init.constant_(self.weights1, 0)

        self.weights2 = nn.Parameter(torch.ones(cfg_.DATASETS.CLASS[1]))
        torch.nn.init.normal_(self.weights2, mean=0, std=1)
        nn.init.constant_(self.weights2, 0)

        self.weights3 = nn.Parameter(torch.ones(cfg_.DATASETS.CLASS[2]))
        torch.nn.init.normal_(self.weights3, mean=0, std=1)
        nn.init.constant_(self.weights3, 0)

        self.weights = nn.Parameter(torch.ones(3))
        torch.nn.init.normal_(self.weights, mean=0, std=1)
        nn.init.constant_(self.weights, 0)

        self.loss_kwargs = loss_kwargs

        self.register_buffer('pixel_mean', torch.Tensor(pixel_mean).view(1, -1, 1, 1), False)
        self.register_buffer('pixel_std', torch.Tensor(pixel_std).view(1, -1, 1, 1), False)

    @classmethod
    def from_config(cls, cfg):
        cfg_ = cfg
        backbone = build_backbone(cfg)
        heads = build_heads(cfg)
        heads_expert1 = build_heads(cfg, num_classes=cfg.DATASETS.CLASS[0])
        heads_expert2 = build_heads(cfg, num_classes=cfg.DATASETS.CLASS[1])
        heads_expert3 = build_heads(cfg, num_classes=cfg.DATASETS.CLASS[2])

        return {
            'cfg_': cfg_,
            'backbone': backbone,
            'heads': heads,
            'heads_expert1': heads_expert1,
            'heads_expert2': heads_expert2,
            'heads_expert3': heads_expert3,
            'pixel_mean': cfg.MODEL.PIXEL_MEAN,
            'pixel_std': cfg.MODEL.PIXEL_STD,
            'loss_kwargs':
                {
                    # loss name
                    'loss_names': cfg.MODEL.LOSSES.NAME,

                    # loss hyperparameters
                    'ce': {
                        'eps': cfg.MODEL.LOSSES.CE.EPSILON,
                        'alpha': cfg.MODEL.LOSSES.CE.ALPHA,
                        'scale': cfg.MODEL.LOSSES.CE.SCALE
                    },
                    'tri': {
                        'margin': cfg.MODEL.LOSSES.TRI.MARGIN,
                        'norm_feat': cfg.MODEL.LOSSES.TRI.NORM_FEAT,
                        'hard_mining': cfg.MODEL.LOSSES.TRI.HARD_MINING,
                        'scale': cfg.MODEL.LOSSES.TRI.SCALE
                    },
                    'circle': {
                        'margin': cfg.MODEL.LOSSES.CIRCLE.MARGIN,
                        'gamma': cfg.MODEL.LOSSES.CIRCLE.GAMMA,
                        'scale': cfg.MODEL.LOSSES.CIRCLE.SCALE
                    },
                    'cosface': {
                        'margin': cfg.MODEL.LOSSES.COSFACE.MARGIN,
                        'gamma': cfg.MODEL.LOSSES.COSFACE.GAMMA,
                        'scale': cfg.MODEL.LOSSES.COSFACE.SCALE
                    }
                }
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        images, data_name = self.preprocess_image(batched_inputs)
        features = self.backbone(images)

        if self.training:
            assert "targets" in batched_inputs, "Person ID annotation are missing in training!"
            targets = batched_inputs["targets"]
            targets_expert = batched_inputs["targets_expert"]

            # PreciseBN flag, When do preciseBN on different dataset, the number of classes in new dataset
            # may be larger than that in the original dataset, so the circle/arcface will
            # throw an error. We just set all the targets to 0 to avoid this problem.
            if targets.sum() < 0: targets.zero_()

            outputs = self.heads(features, targets)
            feat = outputs['features']

            if data_name == "Expert1":
                outputs_exp1_heads = self.heads_expert1(features, targets_expert)          
                outputs_exp1  = outputs_exp1_heads['features']

                self.MB.update(data_name, targets_expert, outputs_exp1)

                # mix style
                # norm_feat = torch.div((feat - torch.mean(feat, dim=1, keepdim=True).detach()), torch.std(feat, dim=1, keepdim=True).detach())

                norm_feat = torch.div((outputs_exp1 - torch.mean(outputs_exp1, dim=1, keepdim=True).detach()), torch.std(outputs_exp1, dim=1, keepdim=True).detach())

                expert1_styles = self.MB.expert_styles("Expert1")
                expert1_new_style = torch.sum(torch.mul(expert1_styles.t().to(self.device), self.weights1), dim=1).detach()
                expert1_new_feat = norm_feat * expert1_new_style[1] + expert1_new_style[0]

                expert2_styles = self.MB.expert_styles("Expert2")
                expert2_new_style = torch.sum(torch.mul(expert2_styles.t().to(self.device), self.weights2), dim=1).detach()
                expert2_new_feat = norm_feat * expert2_new_style[1] + expert2_new_style[0]

                expert3_styles = self.MB.expert_styles("Expert3")
                expert3_new_style = torch.sum(torch.mul(expert3_styles.t().to(self.device), self.weights3), dim=1).detach()
                expert3_new_feat = norm_feat * expert3_new_style[1] + expert3_new_style[0]

                stds = torch.tensor([torch.std(expert1_new_feat), torch.std(expert2_new_feat), torch.std(expert3_new_feat)])
                means = torch.tensor([torch.mean(expert1_new_feat), torch.mean(expert2_new_feat), torch.mean(expert3_new_feat)])

                gama = torch.sum(torch.mul(stds.to(self.device), self.weights))
                beta = torch.sum(torch.mul(means.to(self.device), self.weights))

                mix_feat = gama.detach() * norm_feat + beta.detach()
                
                # outputs['features'] = mix_feat
                
                losses = self.losses(outputs, targets, outputs_exp1_heads, targets_expert)
             
            elif data_name == "Expert2":
                outputs_exp2_heads = self.heads_expert2(features, targets_expert)
                outputs_exp2  = outputs_exp2_heads['features']

                self.MB.update(data_name, targets_expert, outputs_exp2)

                # mix style
                # norm_feat = torch.div((feat - torch.mean(feat, dim=1, keepdim=True).detach()), torch.std(feat, dim=1, keepdim=True).detach())

                norm_feat = torch.div((outputs_exp2 - torch.mean(outputs_exp2, dim=1, keepdim=True).detach()), torch.std(outputs_exp2, dim=1, keepdim=True).detach())

                expert1_styles = self.MB.expert_styles("Expert1")
                expert1_new_style = torch.sum(torch.mul(expert1_styles.t().to(self.device), self.weights1), dim=1).detach()
                expert1_new_feat = norm_feat * expert1_new_style[1] + expert1_new_style[0]

                expert2_styles = self.MB.expert_styles("Expert2")
                expert2_new_style = torch.sum(torch.mul(expert2_styles.t().to(self.device), self.weights2), dim=1).detach()
                expert2_new_feat = norm_feat * expert2_new_style[1] + expert2_new_style[0]

                expert3_styles = self.MB.expert_styles("Expert3")
                expert3_new_style = torch.sum(torch.mul(expert3_styles.t().to(self.device), self.weights3), dim=1).detach()
                expert3_new_feat = norm_feat * expert3_new_style[1] + expert3_new_style[0]

                stds = torch.tensor([torch.std(expert1_new_feat), torch.std(expert2_new_feat), torch.std(expert3_new_feat)])
                means = torch.tensor([torch.mean(expert1_new_feat), torch.mean(expert2_new_feat), torch.mean(expert3_new_feat)])

                gama = torch.sum(torch.mul(stds.to(self.device), self.weights))
                beta = torch.sum(torch.mul(means.to(self.device), self.weights))
                
                mix_feat = gama.detach() * norm_feat + beta.detach()
                
                # outputs['features'] = mix_feat

                outputs_exp2_heads['features'] = mix_feat

                losses = self.losses(outputs, targets, outputs_exp2_heads, targets_expert)

            elif data_name == "Expert3":
                outputs_exp3_heads = self.heads_expert3(features, targets_expert)
                outputs_exp3  = outputs_exp3_heads['features']

                self.MB.update(data_name, targets_expert, outputs_exp3)

                # mix style
                # norm_feat = torch.div((feat - torch.mean(feat, dim=1, keepdim=True).detach()), torch.std(feat, dim=1, keepdim=True).detach())
                
                norm_feat = torch.div((outputs_exp3 - torch.mean(outputs_exp3, dim=1, keepdim=True).detach()), torch.std(outputs_exp3, dim=1, keepdim=True).detach())
                
                expert1_styles = self.MB.expert_styles("Expert1")
                expert1_new_style = torch.sum(torch.mul(expert1_styles.t().to(self.device), self.weights1), dim=1).detach()
                expert1_new_feat = norm_feat * expert1_new_style[1] + expert1_new_style[0]

                expert2_styles = self.MB.expert_styles("Expert2")
                expert2_new_style = torch.sum(torch.mul(expert2_styles.t().to(self.device), self.weights2), dim=1).detach()
                expert2_new_feat = norm_feat * expert2_new_style[1] + expert2_new_style[0]

                expert3_styles = self.MB.expert_styles("Expert3")
                expert3_new_style = torch.sum(torch.mul(expert3_styles.t().to(self.device), self.weights3), dim=1).detach()
                expert3_new_feat = norm_feat * expert3_new_style[1] + expert3_new_style[0]

                stds = torch.tensor([torch.std(expert1_new_feat), torch.std(expert2_new_feat), torch.std(expert3_new_feat)])
                means = torch.tensor([torch.mean(expert1_new_feat), torch.mean(expert2_new_feat), torch.mean(expert3_new_feat)])

                gama = torch.sum(torch.mul(stds.to(self.device), self.weights))
                beta = torch.sum(torch.mul(means.to(self.device), self.weights))

                mix_feat = gama.detach() * norm_feat + beta.detach()
                
                # outputs['features'] = mix_feat

                outputs_exp3_heads['features'] = mix_feat
            
                losses = self.losses(outputs, targets, outputs_exp3_heads, targets_expert)

            return losses

        else:
            outputs = self.heads(features)
            return outputs

    def preprocess_image(self, batched_inputs):
        """
        Normalize and batch the input images.
        """

        if self.training:
            if self.cfg_.DATASETS.NAMES[0][:5].lower() in batched_inputs['img_paths'][0].lower():
                data_name = 'Expert1'
            elif self.cfg_.DATASETS.NAMES[1][:5].lower() in batched_inputs['img_paths'][0].lower():
                data_name = 'Expert2'
            elif self.cfg_.DATASETS.NAMES[2][:5].lower() in batched_inputs['img_paths'][0].lower():
                data_name = 'Expert3'
        else:
            data_name = None

        if isinstance(batched_inputs, dict):
            images = batched_inputs['images']
        elif isinstance(batched_inputs, torch.Tensor):
            images = batched_inputs
        else:
            raise TypeError("batched_inputs must be dict or torch.Tensor, but get {}".format(type(batched_inputs)))

        if not self.eval():
            images.sub_(self.pixel_mean).div_(self.pixel_std)
        return images, data_name

    def losses(self, outputs, gt_labels, outputs_exp, targets_exp):
        """
        Compute loss from modeling's outputs, the loss function input arguments
        must be the same as the outputs of the model forwarding.
        """
        # model predictions
        # fmt: off
        """ glo """
        pred_class_logits = outputs['pred_class_logits'].detach()
        cls_outputs       = outputs['cls_outputs']
        pred_features     = outputs['features']

        """ exp """
        pred_class_logits_exp = outputs_exp['pred_class_logits'].detach()
        cls_outputs_exp       = outputs_exp['cls_outputs']
        pred_features_exp     = outputs_exp['features']

        # fmt: on

        # Log prediction accuracy
        log_accuracy(pred_class_logits, gt_labels)

        loss_dict = {}
        loss_names = self.loss_kwargs['loss_names']

        if 'CrossEntropyLoss' in loss_names:
            ce_kwargs = self.loss_kwargs.get('ce')
            loss_dict['loss_cls_glo'] = cross_entropy_loss(
                cls_outputs,
                gt_labels,
                ce_kwargs.get('eps'),
                ce_kwargs.get('alpha')
            ) * ce_kwargs.get('scale')

            ce_kwargs = self.loss_kwargs.get('ce')
            loss_dict['loss_cls_exp'] = cross_entropy_loss(
                cls_outputs_exp,
                targets_exp,
                ce_kwargs.get('eps'),
                ce_kwargs.get('alpha')
            ) * ce_kwargs.get('scale')

        if 'TripletLoss' in loss_names:
            tri_kwargs = self.loss_kwargs.get('tri')
            loss_dict['loss_triplet_glo'] = triplet_loss(
                pred_features,
                gt_labels,
                tri_kwargs.get('margin'),
                tri_kwargs.get('norm_feat'),
                tri_kwargs.get('hard_mining')
            ) * tri_kwargs.get('scale')

            tri_kwargs = self.loss_kwargs.get('tri')
            loss_dict['loss_triplet_exp'] = triplet_loss(
                pred_features_exp,
                targets_exp,
                tri_kwargs.get('margin'),
                tri_kwargs.get('norm_feat'),
                tri_kwargs.get('hard_mining')
            )

        if 'CircleLoss' in loss_names:
            circle_kwargs = self.loss_kwargs.get('circle')
            loss_dict['loss_circle'] = pairwise_circleloss(
                pred_features,
                gt_labels,
                circle_kwargs.get('margin'),
                circle_kwargs.get('gamma')
            ) * circle_kwargs.get('scale')

        if 'Cosface' in loss_names:
            cosface_kwargs = self.loss_kwargs.get('cosface')
            loss_dict['loss_cosface'] = pairwise_cosface(
                pred_features,
                gt_labels,
                cosface_kwargs.get('margin'),
                cosface_kwargs.get('gamma'),
            ) * cosface_kwargs.get('scale')

    #    loss_dict['loss_content'] = content_loss(content[0], content[2]) + content_loss(content[1], content[2])

        return loss_dict
    
    def get_bn_list(self):
        mean_expert1 = None
        var_expert1 = None
        for name, value in self.heads_expert1.state_dict().items():
            if "running_mean" in name:
                mean_expert1 = torch.mean(value)
            elif "running_var" in name:
                var_expert1 = torch.mean(value)
                
        mean_expert2 = None
        var_expert2 = None
        for name, value in self.heads_expert2.state_dict().items():
            if "running_mean" in name:
                mean_expert2 = torch.mean(value)
            elif "running_var" in name:
                var_expert2 = torch.mean(value)

        mean_expert3 = None
        var_expert3 = None
        for name, value in self.heads_expert3.state_dict().items():
            if "running_mean" in name:
                mean_expert3 = torch.mean(value)
            elif "running_var" in name:
                var_expert3 = torch.mean(value)

        mean = torch.mean(mean_expert1 + mean_expert2 + mean_expert3)
        std = torch.mean(var_expert1 + var_expert2 + var_expert3).sqrt()

        return mean, std
    
def content_loss(feat, target):
    loss = torch.nn.MSELoss()
    return loss(feat, target)
    