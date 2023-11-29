# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch
import torch.nn.functional as F
from torch import nn

from fastreid.config import configurable
from fastreid.layers import *
from fastreid.layers import pooling, any_softmax
from fastreid.layers.weight_init import weights_init_kaiming
from .build import REID_HEADS_REGISTRY


@REID_HEADS_REGISTRY.register()
class EmbeddingAggHead(nn.Module):
    """
    EmbeddingAggHead perform all feature aggregation in an embedding task, such as reid, image retrieval
    and face recognition

    It typically contains logic to

    1. feature aggregation via global average pooling and generalized mean pooling
    2. (optional) batchnorm, dimension reduction and etc.
    2. (in training only) margin-based softmax logits computation

    使用两个embedding head, 分别提取域无关特征和域相关特征, 之后融合特征(sum or cat)
    """

    @configurable
    def __init__(
            self,
            *,
            feat_dim,
            embedding_dim,
            num_classes,
            neck_feat,
            pool_type,
            cls_type,
            scale,
            margin,
            with_bnneck,
            norm_type,
            agg_type
    ):
        """
        NOTE: this interface is experimental.

        Args:
            feat_dim:
            embedding_dim:
            num_classes:
            neck_feat:
            pool_type:
            cls_type:
            scale:
            margin:
            with_bnneck:
            norm_type:
            agg_type:
        """
        super().__init__()

        # Pooling layer
        assert hasattr(pooling, pool_type), "Expected pool types are {}, " \
                                            "but got {}".format(pooling.__all__, pool_type)
        self.pool_layer = getattr(pooling, pool_type)()

        self.neck_feat = neck_feat

        self.agg_type = agg_type

        neck = []

        if embedding_dim > 0:
            neck.append(nn.Conv2d(feat_dim, embedding_dim, 1, 1, bias=False))
            feat_dim = embedding_dim

        if with_bnneck:
            neck.append(get_norm(norm_type, feat_dim, bias_freeze=True))
        
        self.bottleneck1 = nn.Sequential(get_norm('BN', feat_dim, bias_freeze=True))
        self.bottleneck2 = nn.Sequential(nn.InstanceNorm2d(feat_dim, affine=True))

        # Classification head
        assert hasattr(any_softmax, cls_type), "Expected cls types are {}, " \
                                               "but got {}".format(any_softmax.__all__, cls_type)
        if agg_type == 'cat':
            feat_dim = 4096
        self.weight = nn.Parameter(torch.Tensor(num_classes, feat_dim))
        self.cls_layer = getattr(any_softmax, cls_type)(num_classes, scale, margin)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.bottleneck1.apply(weights_init_kaiming)
        self.bottleneck2.apply(weights_init_kaiming)
        nn.init.normal_(self.weight, std=0.01)

    @classmethod
    def from_config(cls, cfg):
        # fmt: off
        feat_dim      = cfg.MODEL.BACKBONE.FEAT_DIM
        embedding_dim = cfg.MODEL.HEADS.EMBEDDING_DIM
        num_classes   = cfg.MODEL.HEADS.NUM_CLASSES
        neck_feat     = cfg.MODEL.HEADS.NECK_FEAT
        pool_type     = cfg.MODEL.HEADS.POOL_LAYER
        cls_type      = cfg.MODEL.HEADS.CLS_LAYER
        scale         = cfg.MODEL.HEADS.SCALE
        margin        = cfg.MODEL.HEADS.MARGIN
        with_bnneck   = cfg.MODEL.HEADS.WITH_BNNECK
        norm_type     = cfg.MODEL.HEADS.NORM
        agg_type      = cfg.MODEL.HEADS.AGG_TYPE
        # fmt: on
        return {
            'feat_dim': feat_dim,
            'embedding_dim': embedding_dim,
            'num_classes': num_classes,
            'neck_feat': neck_feat,
            'pool_type': pool_type,
            'cls_type': cls_type,
            'scale': scale,
            'margin': margin,
            'with_bnneck': with_bnneck,
            'norm_type': norm_type,
            'agg_type': agg_type,
        }

    def forward(self, features, targets=None):
        """
        See :class:`ReIDHeads.forward`.
        """
        pool_feat = self.pool_layer(features)

        neck_feat1 = self.bottleneck1(pool_feat)
        neck_feat1= neck_feat1[..., 0, 0]

        neck_feat2 = self.bottleneck2(pool_feat)
        neck_feat2 = neck_feat2[..., 0, 0]

        # 融合特征：sum or cat
        if self.agg_type == 'sum':
            neck_feat = neck_feat1 + neck_feat2
        if self.agg_type == 'cat':
            neck_feat = torch.cat((neck_feat1, neck_feat2), 1)


        # Evaluation
        # fmt: off
        if not self.training: return neck_feat
        # fmt: on

        # Training
        if self.cls_layer.__class__.__name__ == 'Linear':
            logits = F.linear(neck_feat, self.weight)
        else:
            logits = F.linear(F.normalize(neck_feat), F.normalize(self.weight))

        # Pass logits.clone() into cls_layer, because there is in-place operations
        cls_outputs = self.cls_layer(logits.clone(), targets)

        # fmt: off
        if self.neck_feat == 'before':  feat = pool_feat[..., 0, 0]
        elif self.neck_feat == 'after': feat = neck_feat
        else:                           raise KeyError(f"{self.neck_feat} is invalid for MODEL.HEADS.NECK_FEAT")
        # fmt: on

        return {
            "cls_outputs": cls_outputs,
            "pred_class_logits": logits.mul(self.cls_layer.s),
            "features": feat,
        }
