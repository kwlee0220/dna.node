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


class BDDDataset(data.Dataset):
    """
    MCMOT Dataset Class
    """
    def __init__(self,
                 path="data/bdd100k",
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
        self.path = path
        self.dataset, self.pairs, self.pairs_images, self.annotations = self.get_datasets(path)
        self.dataset_idxes = list(self.dataset.keys())
        self.transforms = transforms

    def get_datasets(self, path):
        """
        :param path: annotation(json format) path
        :return: dataset, indexed
        """
        json_file = json.load(open(os.path.join(path, "annotation/bdd100k_track_train_coco.json")))

        annotations = json_file['annotations']

        image_to_video_id = {}
        for img in json_file['images']:
            image_to_video_id[img['id']] = img['video_id']

        image_annotations = {}
        tracked_objs = {}
        for anno in annotations:
            if anno['category_id'] == 3:
                video_id = image_to_video_id[anno['image_id']]
                track_id = video_id * 10000 + anno['instance_id']
                anno['instance_id'] = track_id
                anno['category_id'] = 1
                if track_id not in tracked_objs:
                    tracked_objs[track_id] = []
                tracked_objs[track_id].append(anno)

                image_id = anno['image_id']
                if image_id not in image_annotations:
                    image_annotations[image_id] = []
                image_annotations[image_id].append(anno)

        tracked_pair_images = {}
        for obj_id in tracked_objs:
            for anno in tracked_objs[obj_id]:
                if anno['image_id'] not in tracked_pair_images:
                    tracked_pair_images[anno['image_id']] = []
                tracked_pair_images[anno['image_id']].append(obj_id)

        # image_ids = list(tracked_pair_images.keys())

        images = json_file['images']
        image_id_dict = {}
        for img in images:
            image_id_dict[img['id']] = img

        id_images = {}
        for img_id in tqdm.tqdm(tracked_pair_images):
            # if img['id'] in image_ids:
            id_images[img_id] = image_id_dict[img_id]

        paired_tracked_images = {}
        for t in tracked_objs:
            paired_tracked_images[t] = [a['image_id'] for a in tracked_objs[t]]

        return id_images, tracked_pair_images, paired_tracked_images, image_annotations

    def __getitem__(self, index):
        """
        :param index:
        :return:
        """
        image_id = self.dataset_idxes[index]

        tracked = self.pairs[image_id]
        track_id = random.choice(tracked)

        image_list = self.pairs_images[track_id]

        frames = [image_id, random.choice(image_list)]

        videos = []
        instances = []
        for frame in frames:
            image = self.dataset[frame]
            img = cv2.imread(os.path.join("data/bdd100k/data/images/track/train", image['file_name']))
            f_h, f_w, c = img.shape
            image_size = (f_h, f_w)
            annotations = self.annotations[frame]

            instance = Instances(image_size)
            bboxes = []
            ids = []
            categories = []
            for anno in annotations:
                xywh = anno['bbox']
                xyxy = [xywh[0], xywh[1], xywh[2] + xywh[0], xywh[3] + xywh[1]]
                bboxes.append(xyxy)

                ids.append(anno['instance_id'])
                categories.append(anno['category_id'])

            instance.set("gt_boxes", Boxes(bboxes))
            instance.set("ids", torch.Tensor(ids).to(torch.long))
            instance.set("gt_classes", torch.Tensor(categories).to(torch.long))

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            videos.append(img)
            instances.append(instance)

        # augmentation
        if self.transforms is not None:
            videos, instances = self.transforms(videos, instances)

        outputs = []
        for idx in range(len(videos)):
            outputs.append({
                "height": f_h,
                "width": f_w,
                "image": videos[idx],
                "instances": instances[idx]
            })
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

            ids.append(anno['track_id'])
            categories.append(anno['category_id'])

        instance.set("gt_boxes", Boxes(bboxes))
        instance.set("ids", torch.Tensor(ids).to(torch.long))
        instance.set("gt_classes", torch.Tensor(categories).to(torch.long))
        return instance

    def __len__(self):
        return len(self.dataset)