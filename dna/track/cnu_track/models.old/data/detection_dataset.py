'''
@Project ：ETRI
@File ：mcmot_dataset.py
@Author ： Hyunseop Kim
@Date ：22. 11. 22.
'''
import torch
from torch.utils.data.dataset import ConcatDataset

import itertools

import json
import os

import random
import numpy as np
import torch.utils.data as data
import cv2
from PIL import Image
import tqdm

from models.structures import Instances, Boxes


class Images():
    """
    Images Class
    """
    def __init__(self, path, video, image, annotations, track):
        """
        init function
        """
        self.video = video
        self.image = image
        self.annotations = annotations
        self.track = track
        self.file_name = os.path.join(path, self.image['file_name'])


class DetectionDataset(data.Dataset):
    """
    MCMOT Dataset Class
    """
    def __init__(self,
                 path="data/AIcity/annotations",
                 is_train=True,
                 transforms=None
                 ):
        """
        :param path: annotation file path
        :param is_train:
        :param transforms: for augmentation
        """
        self.train = is_train

        print("initialize datasets")
        self.dataset = self.get_datasets(path)
        self.transforms = transforms

        if "bdd100k" in path:
            self.path = os.path.join(path, "train2017")
        else:
            self.path = path

    def get_datasets(self, path):
        """
        :param path: annotation(json format) path
        :return: dataset, indexed
        """
        json_file = json.load(open(os.path.join(path, "annotations/train.json")))

        maps = []
        annotations = {}
        if "bdd100k" in path:
            for anno in json_file['annotations']:
                if anno['category_id'] == 2 or anno['category_id'] == 4 or anno['category_id'] == 5:
                    image_id = anno['image_id']
                    if image_id not in annotations:
                        annotations[image_id] = []
                    anno['category_id'] = 1
                    annotations[image_id].append(anno)
        else:
            for anno in json_file['annotations']:
                image_id = anno['image_id']
                if image_id not in annotations:
                    annotations[image_id] = []
                annotations[image_id].append(anno)

        for image in json_file['images']:
            image_id = image['id']
            if image_id in annotations:
                maps.append({
                    "image": image,
                    "instances": annotations[image_id]
                })

        return maps

    def __getitem__(self, index):
        """
        :param index:
        :return:
        """
        data = self.dataset[index]

        frame = data['image']
        annotations = data['instances']

        img = cv2.imread(os.path.join(self.path, frame['file_name']))
        f_h, f_w, c = img.shape
        image_size = (f_h, f_w)
        instance = self.annotation_to_instance(annotations, image_size)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)

        # augmentation
        if self.transforms is not None:
            videos, instances = self.transforms([img], [instance])

        outputs = [{
            "height": f_h,
            "width": f_w,
            "image": videos[0],
            "instances": instances[0]
        }]
        return outputs

    def annotation_to_instance(self, annotations, image_size):
        """
        annotation_to_instance
        """
        instance = Instances(image_size)
        bboxes = []
        ids = []
        categories = []
        for anno in annotations:
            xywh = anno['bbox']
            xyxy = [xywh[0], xywh[1], xywh[2] + xywh[0], xywh[3] + xywh[1]]
            bboxes.append(xyxy)
            if "track_id" in anno:
                ids.append(anno['track_id'])
            else:
                ids.append(anno['id'])
            categories.append(anno['category_id'])

        instance.set("gt_boxes", Boxes(bboxes))
        instance.set("ids", torch.Tensor(ids).to(torch.long))
        instance.set("gt_classes", torch.Tensor(categories).to(torch.long))
        return instance

    def __len__(self):
        return len(self.dataset)


class VideoDatasetBatchCollator(object):
    """
    From a list of samples from the dataset,
    returns the batched images and targets.
    This should be passed to the DataLoader
    """

    def __init__(self, size_divisible=0):
        self.size_divisible = size_divisible

    def __call__(self, batch):
        return list(itertools.chain(*batch))


from detectron2.utils.comm import get_world_size
from mcmot.data.adapters.augmentation.build_augmentation import build_siam_augmentation
from detectron2.data.samplers.distributed_sampler import TrainingSampler


def build_train_data_loader(data_path, cfg, shuffle=True):
    """
    build_train_data_loader
    """
    num_gpus = get_world_size()
    dataset_list = cfg.DATASETS.TRAIN
    batch_size = cfg.SOLVER.VIDEO_CLIPS_PER_BATCH

    transforms = build_siam_augmentation(cfg, is_train=True)

    if len(data_path) == 1:
        dataset = DetectionDataset(data_path[0],
                                   transforms=transforms)
    else:
        datasets = []
        for p in data_path:
            dataset = DetectionDataset(p, transforms=transforms)
            datasets.append(dataset)
        dataset = ConcatDataset(datasets)

    sampler = TrainingSampler(len(dataset), shuffle=shuffle)
    batch_sampler = torch.utils.data.sampler.BatchSampler(
            sampler, batch_size, drop_last=True
    )

    num_workers = cfg.DATALOADER.NUM_WORKERS
    collator = VideoDatasetBatchCollator(cfg.DATALOADER.SIZE_DIVISIBILITY)
    data_loader = torch.utils.data.DataLoader(dataset,
                                              num_workers=num_workers,
                                              batch_sampler=batch_sampler,
                                              collate_fn=collator)
    return data_loader