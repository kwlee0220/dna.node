#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：ETRI
@File ：config.py
@Author ： Hyunseop Kim
@Date ：22. 11. 22.
'''
from fvcore.common.config import CfgNode as CN

def add_configs():

    _C = CN()
    _C.MODEL = CN()
    _C.MODEL.PIXEL_MEAN = [103.530, 116.280, 123.675]
    _C.MODEL.PIXEL_STD = [1.0, 1.0, 1.0]
    _C.MODEL.DEVICE = 'cuda'
    _C.MODEL.FPN = CN()

    _C.MODEL.FPN.IN_FEATURES = ["dla3", "dla4", "dla5"]

    _C.MODEL.BIFPN = CN()
    _C.MODEL.BIFPN.NUM_LEVELS = 3
    _C.MODEL.BIFPN.NUM_BIFPN = 4
    _C.MODEL.BIFPN.NORM = 'GN'
    _C.MODEL.BIFPN.OUT_CHANNELS = 160
    _C.MODEL.BIFPN.SEPARABLE_CONV = False

    _C.MODEL.DLA = CN()
    _C.MODEL.DLA.OUT_FEATURES = ['dla2']
    _C.MODEL.DLA.USE_DLA_UP = True
    _C.MODEL.DLA.NUM_LAYERS = 34
    _C.MODEL.DLA.MS_OUTPUT = False
    _C.MODEL.DLA.NORM = 'BN'
    _C.MODEL.DLA.DLAUP_IN_FEATURES = ['dla3', 'dla4', 'dla5']
    _C.MODEL.DLA.DLAUP_NODE = 'conv'

    _C.MODEL.TRACK_HEAD = CN()
    _C.MODEL.TRACK_HEAD.POOLER_SCALES = (0.125,)

    _C.INPUT = CN()
    _C.INPUT.MIN_SIZE_TRAIN = (800,)
    _C.INPUT.MIN_SIZE_TRAIN_SAMPLING = "choice"
    _C.INPUT.MAX_SIZE_TRAIN = 1333
    _C.INPUT.MIN_SIZE_TEST = 800
    _C.INPUT.MAX_SIZE_TEST = 1333
    _C.INPUT.RANDOM_FLIP = "horizontal"

    _C.INPUT.FORMAT = "BGR"
    _C.INPUT.MOTION_LIMIT = 0.1
    _C.INPUT.COMPRESSION_LIMIT = 50
    _C.INPUT.MOTION_BLUR_PROB = 0.5
    _C.INPUT.AMODAL = False
    _C.INPUT.BRIGHTNESS = 0.1
    _C.INPUT.CONTRAST = 0.1
    _C.INPUT.SATURATION = 0.1
    _C.INPUT.HUE = 0.1

    _C.DATALOADER = CN()
    _C.DATALOADER.NUM_WORKERS = 4
    _C.DATALOADER.ASPECT_RATIO_GROUPING = True
    _C.DATALOADER.SAMPLER_TRAIN = "TrainingSampler"
    _C.DATALOADER.REPEAT_THRESHOLD = 0.0
    _C.DATALOADER.FILTER_EMPTY_ANNOTATIONS = True
    _C.DATALOADER.SIZE_DIVISIBILITY = 32
    return _C.clone()