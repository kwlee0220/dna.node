#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：ETRI
@File ：embedding_track.py
@Author ： https://github.com/SysCV/qdtrack
@Date ：22. 11. 22.
'''

import torch
from torch import nn
import torch.nn.functional as F

# from detectron2.layers import Conv2d
from dna.track.qdtrack.models.layers.wrappers import Conv2d
from .embedding_losses import MultiPosCrossEntropyLoss
from .l2_loss import L2Loss


class EmbedHead(nn.Module):
    """ Embedding head """
    def __init__(self,
                 num_convs=4,
                 num_fcs=2,
                 roi_feat_size=7,
                 in_channels=256,
                 conv_out_channels=256,
                 fc_out_channels=512,
                 embed_channels=2048,
                 softmax_temp=-1,
                 loss_weight=1.,
                 device='cuda'):
        """
        :param num_convs: embedding conv block number
        :param num_fcs: embedding fc block number
        :param roi_feat_size: roi feature size
        :param in_channels: input channel dim
        :param conv_out_channels: convolutionl layer output channel dim
        :param fc_out_channels: fc output chnnel dim
        :param embed_channels: ReID embedding vector dim
        :param softmax_temp:
        :param loss_weight:
        :param device:
        """
        super(EmbedHead, self).__init__()
        self.device = device
        self.num_convs = num_convs
        self.num_fcs = num_fcs
        self.in_channels = in_channels
        self.fc_out_channels = fc_out_channels
        self.roi_feat_size = roi_feat_size
        self.conv_out_channels = conv_out_channels
        self.softmax_temp = softmax_temp

        self.relu = F.relu

        self.convs, self.fcs, last_layer_dim = self._add_conv_fc_branch(
            self.num_convs, self.num_fcs, self.in_channels)
        self.fc_embed = nn.Linear(last_layer_dim, embed_channels)

        self.loss_track = MultiPosCrossEntropyLoss(
            loss_weight=0.2)  # 추후 config로 설정가능하게 바꾸기
        self.loss_track_aux = L2Loss(neg_pos_ub=3,
                                     pos_margin=0.7,
                                     neg_margin=0.3,
                                     hard_mining=True,
                                     reduction='mean',
                                     loss_weight=1.0)
        self.loss_weight = loss_weight

    def _add_conv_fc_branch(self, num_convs, num_fcs, in_channels=256):
        """ Generate convlution layers and fc layer for embedding head """
        last_layer_dim = in_channels

        # add branch specific conv layers
        convs = nn.ModuleList()
        if num_convs > 0:
            for i in range(num_convs):
                conv_in_channels = (
                    last_layer_dim if i == 0 else self.conv_out_channels)
                convs.append(Conv2d(in_channels=conv_in_channels,
                                    out_channels=self.conv_out_channels,
                                    kernel_size=3,
                                    padding=1,
                                    norm=None, activation=None))
            last_layer_dim = self.conv_out_channels

        # add branch specific fc layers
        fcs = nn.ModuleList()
        if num_fcs > 0:
            last_layer_dim *= (self.roi_feat_size * self.roi_feat_size)
            for i in range(num_fcs):
                fc_in_channels = (
                    last_layer_dim if i == 0 else self.fc_out_channels)
                fcs.append(nn.Linear(in_features=fc_in_channels,
                                     out_features=self.fc_out_channels))
            last_layer_dim = self.fc_out_channels
        return convs, fcs, last_layer_dim

    def get_track_targets(self, gt_match_indices, key_sampling_results,
                          ref_sampling_results):
        """
         Get ground truth for embedding head training
         return matched instance between key instance and reference instances
        """
        track_targets = []
        track_weights = []
        for _gt_match_indices, key_res, ref_res in zip(gt_match_indices,
                                                       key_sampling_results,
                                                       ref_sampling_results):
            targets = _gt_match_indices.new_zeros(
                (key_res.pos_bboxes.size(0), ref_res.bboxes.size(0)),
                dtype=torch.int)
            _match_indices = _gt_match_indices[key_res.pos_assigned_gt_inds]
            pos2pos = (_match_indices.view(
                -1, 1) == ref_res.pos_assigned_gt_inds.view(1, -1)).int()
            targets[:, :pos2pos.size(1)] = pos2pos
            weights = (targets.sum(dim=1) > 0).float()
            track_targets.append(targets)
            track_weights.append(weights)
        return track_targets, track_weights

    def get_target(self, key_matches_indices, ref_matches_indices, key_nums,
                   ref_nums):
        """
         Get ground truth for embedding head training
         return matched instance between key instance and reference instances
        """
        track_targets = []
        track_weights = []

        for key_res, ref_res, key_num, ref_num in zip(key_matches_indices,
                                                      ref_matches_indices,
                                                      key_nums, ref_nums):
            embed_targets = torch.zeros(
                (key_num, ref_num))  # , dtype=torch.float)
            pos2pos = (key_res.view(-1, 1) == ref_res.view(1, -1))  # .float()
            embed_targets[:, :pos2pos.size(1)] = pos2pos
            weights = (embed_targets.sum(dim=1) > 0).float()
            track_targets.append(embed_targets.to(self.device))
            track_weights.append(weights.to(self.device))

        return track_targets, track_weights

    def match(self, key_embeds, ref_embeds):
        """
         Calculate similarity between key embedding and reference embedding
        """
        assert len(key_embeds) == len(ref_embeds)
        # match tao_embed
        dists, cos_dists = [], []
        for (key_embed, ref_embed) in zip(key_embeds, ref_embeds):
            dist = cal_similarity(key_embed,
                                  ref_embed,
                                  method='dot_product',
                                  temperature=self.softmax_temp)
            dists.append(dist)
            if self.loss_track_aux is not None:
                cos_dist = cal_similarity(key_embed, ref_embed, method='cosine')
                cos_dists.append(cos_dist)
            else:
                cos_dists.append(None)
        return dists, cos_dists

    def loss(self, dists, cos_dists, targets, weights):
        """
         Calculate embedding loss
        """
        losses = dict()

        loss_track = 0.
        loss_track_aux = 0.
        # num_loss = 0.
        for _dists, _cos_dists, _targets, _weights in zip(
                dists, cos_dists, targets, weights):
            avg_factor = _weights.sum()
            if avg_factor == 0:
                continue
            loss_track += self.loss_track(
                _dists, _targets, _weights, avg_factor=avg_factor)
            if self.loss_track_aux is not None:
                loss_track_aux += self.loss_track_aux(_cos_dists, _targets)
        losses['loss_track'] = loss_track / len(dists)

        if self.loss_track_aux is not None:
            losses['loss_track_aux'] = loss_track_aux / len(dists)

        return losses

    def forward(self, x):
        """
        :param x: RoI features
        :return: embedding vectors
        """
        if self.num_convs > 0:
            for i, conv in enumerate(self.convs):
                x = conv(x)
        x = torch.flatten(x, start_dim=1)
        if self.num_fcs > 0:
            for i, fc in enumerate(self.fcs):
                x = self.relu(fc(x))
        x = self.fc_embed(x)
        return x


def cal_similarity(key_embeds,
                   ref_embeds,
                   method='dot_product',
                   temperature=-1):
    """
    :param key_embeds: Key instance embedding vector
    :param ref_embeds: reference instance embedding vector
    :param method: cosine or dot-product distance
    :param temperature:
    :return: similarity distance
    """
    assert method in ['dot_product', 'cosine']

    if key_embeds.size(0) == 0 or ref_embeds.size(0) == 0:
        return torch.zeros((key_embeds.size(0), ref_embeds.size(0)),
                           device=key_embeds.device)

    if method == 'cosine':
        # cosine similarity
        key_embeds = F.normalize(key_embeds, p=2, dim=1)
        ref_embeds = F.normalize(ref_embeds, p=2, dim=1)
        return torch.mm(key_embeds, ref_embeds.t())
    elif method == 'dot_product':
        # dot-product similarity
        if temperature > 0:
            dists = cal_similarity(key_embeds, ref_embeds, method='cosine')
            dists /= temperature
            return dists
        else:
            return torch.mm(key_embeds, ref_embeds.t())
