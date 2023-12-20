#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：ETRI
@File ：rcnn.py
@Author ： Hyunseop Kim
@Date ：22. 11. 22.
'''

import torch
from torch import nn
import torch.nn.functional as F

from ..structures import ImageList, Instances, Boxes
from ..backbone.bifpn_fcos import build_p35_fcos_dla_bifpn_backbone

from .poolers import ROIPooler
from .tracker.embedding_track import EmbedHead


class MOT(nn.Module):
    """
    Detectron2-based custom META_ARCHITECTURE
    """
    def __init__(self, cfg):
        """
        :param cfg: Detectron2 config file
        """
        super(MOT, self).__init__()
        self.cfg = cfg

        pixel_mean = cfg.MODEL.PIXEL_MEAN
        pixel_std = cfg.MODEL.PIXEL_STD

        self.register_buffer("pixel_mean",
                             torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std",
                             torch.tensor(pixel_std).view(-1, 1, 1), False)

        self.device = torch.device(cfg.MODEL.DEVICE)
        self.backbone = build_p35_fcos_dla_bifpn_backbone(cfg)

        pooler_size = 32
        self.track_roi_extractor = ROIPooler(
            output_size=(pooler_size, pooler_size),
            scales=cfg.MODEL.TRACK_HEAD.POOLER_SCALES,
            sampling_ratio=0,
            pooler_type="ROIAlignV2"
        )

        self.embedhead = EmbedHead(
            num_fcs=2,
            embed_channels=1024,
            in_channels=160,
            roi_feat_size=pooler_size,
            softmax_temp=-1,
            device=cfg.MODEL.DEVICE
        )

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

    def inference(self, batched_inputs):
        images = self.preprocess_image(batched_inputs)
        instances = [x["instances"].to(self.device) for x in batched_inputs]
        results = []

        for i in instances:
            i.set("pred_boxes", i.gt_boxes)
            i.set("pred_logits", i.scores)
            i.set("pred_classes", i.gt_classes)
            results.append(i)

        features = self.backbone(images.tensor)
        features = [features[f] for f in features]
        f_len = len(self.cfg.MODEL.TRACK_HEAD.POOLER_SCALES)
        features = features[:f_len]

        roi_features = self.track_roi_extractor(features, [b.pred_boxes for b in results])
        reid = self.embedhead(roi_features)
        return F.normalize(reid, p=2, dim=1)

    def forward(self, batched_inputs):
        """
        :param batched_inputs: network input
        {
        "instances": ground truth instances
        "image": input image
        }
        :return: detection or tracking predicted instances
        """
        if not self.training:
            return self.inference(batched_inputs)


def build_models(cfg):
    return MOT(cfg)
