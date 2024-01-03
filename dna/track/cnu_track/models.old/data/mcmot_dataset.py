'''
@Project ：ETRI
@File ：mcmot_dataset.py
@Author ： Hyunseop Kim
@Date ：22. 11. 22.
'''
import torch
import itertools

import json
import os

import random
import torch.utils.data as data
import cv2
from PIL import Image

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


class MCMOTDataset(data.Dataset):
    """
    MCMOT Dataset Class
    """
    def __init__(self,
                 path="data/AIcity/annotations",
                 is_train=True,
                 transforms=None,
                 validation=False,
                 ):
        """
        :param path: annotation file path
        :param is_train:
        :param transforms: for augmentation
        """
        self.train = is_train

        print("initialize datasets")
        self.path = path
        if validation:
            self.dataset, self.pairs, self.pairs_images, self.annotations = self.get_datasets(path, annotation="validation.json")
        else:
            self.dataset, self.pairs, self.pairs_images, self.annotations = self.get_datasets(path)
        self.dataset_idxes = list(self.dataset.keys())
        self.transforms = transforms

    def get_datasets(self, path, annotation="train.json"):
        """
        :param path: annotation(json format) path
        :return: dataset, indexed
        """
        json_file = json.load(open(os.path.join(path, "annotations/{}".format(annotation))))

        annotations = json_file['annotations']

        image_annotations = {}
        tracked_objs = {}
        for anno in annotations:
            track_id = anno['track_id']
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

        image_ids = list(tracked_pair_images.keys())

        images = json_file['images']
        id_images = {}
        for img in images:
            if img['id'] in image_ids:
                id_images[img['id']] = img

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
            img = cv2.imread(os.path.join(self.path, image['file_name']))
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

                ids.append(anno['track_id'])
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