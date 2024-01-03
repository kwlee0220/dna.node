'''
@Project ：ETRI
@File ：video_augmentation.py
@Author ： Hyunseop Kim
@Date ：22. 11. 22.
'''

import torch
import random
import numpy as np
from torchvision.transforms import ColorJitter as ImageColorJitter

from .image_augmentation \
    import \
    ImageResize, \
    ImageMotionBlur, \
    ImageCompression

from ..transform import Resize, ResizeTransform, HFlipTransform
from ..transform import transform as T
from dna.track.qdtrack.models.structures import Boxes, Instances


class VideoTransformer(object):
    """
    Video Transformer class
    """
    def __init__(self, transform_fn=None):
        if transform_fn is None:
            raise KeyError('Transform function should not be None.')
        self.transform_fn = transform_fn

    def __call__(self, video, target=None):
        """
        A data transformation wrapper for video
        :param video: a list of images
        :param target: a list of BoxList (per image)
        """
        if not isinstance(video, (list, tuple)):
            return self.transform_fn(video, target)

        new_video = []
        new_target = []
        for (image, image_target) in zip(video, target):
            (image, image_target) = self.transform_fn(image, image_target)
            new_video.append(image)
            new_target.append(image_target)

        return new_video, new_target


class SiamVideoResize(ImageResize):
    """
    SiamVideoResize Class
    """
    def __init__(self, min_size, max_size, size_divisibility):
        super(SiamVideoResize, self).__init__(min_size, max_size,
                                              size_divisibility)

    def __call__(self, video, target=None):
        if not isinstance(video, (list, tuple)):
            return super(SiamVideoResize, self).__call__(video, target)

        assert len(video) >= 1
        new_size = self.get_size(video[0].shape[:2])
        new_video = []
        new_target = []
        for (image, image_target) in zip(video, target):
            (image, image_target) = self._resize(image, new_size, image_target)
            new_video.append(np.array(image))
            new_target.append(image_target)

        return new_video, new_target

    def _resize(self, image, size, target=None):
        """
        resize
        """
        resize = [Resize(size)]
        image, transform = T.apply_augmentations(resize, image)
        if target is None:
            return image, target
        transform_box = transform.apply_box(target.gt_boxes.tensor)
        h, w = size
        new_target = Instances((h, w))
        for field in target.get_fields():
            if field == "gt_boxes":
                new_target.set(field, Boxes(transform_box))
            else:
                new_target.set(field, target.get(field))
        return image, new_target


class SiamVideoRandomHorizontalFlip(object):
    """
    SiamVideoRandomHorizontalFlip Class
    """
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, video, target=None):

        if not isinstance(video, (list, tuple)):
            return video, target

        new_video = []
        new_target = []
        # All frames should have the same flipping operation
        if random.random() < self.prob:
            h, w = video[0].shape[:2]
            hflip = [HFlipTransform(w)]
            for (image, image_target) in zip(video, target):
                new_image, transform = T.apply_augmentations(hflip, image)
                transform_boxes = transform.apply_box(
                    image_target.gt_boxes.tensor)
                image_target.set("gt_boxes", Boxes(transform_boxes))
                new_video.append(new_image)
                new_target.append(image_target)
        else:
            new_video = video
            new_target = target
        return new_video, new_target


class SiamVideoColorJitter(ImageColorJitter):
    """
    SiamVideoColorJitter Class
    """
    def __init__(self,
                 brightness=None,
                 contrast=None,
                 saturation=None,
                 hue=None):
        super(SiamVideoColorJitter, self).__init__(brightness, contrast,
                                                   saturation, hue)

    def __call__(self, video, target=None):
        # Color jitter only applies for Siamese Training
        if not isinstance(video, (list, tuple)):
            return video, target

        idx = random.choice((0, 1))
        transform = self.get_params(self.brightness, self.contrast,
                                    self.saturation, self.hue)
        new_video = []
        new_target = []
        for i, (image, image_target) in enumerate(zip(video, target)):
            if i == idx:
                image = self(image)[0]
            new_video.append(image)
            new_target.append(image_target)

        return new_video, new_target


class SiamVideoMotionAugment(object):
    """
    SiamVideoMotionAugment Class
    """
    def __init__(self, motion_limit=None, amodal=False):
        # maximum motion augmentation
        self.motion_limit = min(0.1, motion_limit)
        if motion_limit is None:
            self.motion_limit = 0
        # self.motion_augment = ImageCropResize(self.motion_limit, amodal)
        self.crop_limit = motion_limit
        self.amodal = amodal

    def crop_motion(self, image, target):
        """
        crop motion
        """
        w, h = image.size

        tl_x = int(w * (random.random() * self.crop_limit))
        tl_y = int(h * (random.random() * self.crop_limit))
        br_x = int(w - w * (random.random() * self.crop_limit))
        # keep aspect ratio
        br_y = int((h / w) * (br_x - tl_x) + tl_y)

        if len(target) > 0:
            box = target.gt_boxes.tensor.clone()
            # get the visible part of the objects
            box_w = box[:, 2].clamp(min=0, max=w - 1) - \
                    box[:, 0].clamp(min=0, max=w - 1)
            box_h = box[:, 3].clamp(min=0, max=h - 1) - \
                    box[:, 1].clamp(min=0, max=h - 1)
            box_area = box_h * box_w
            max_area_idx = torch.argmax(box_area, dim=0)
            max_motion_limit_w = int(box_w[max_area_idx] * 0.25)
            max_motion_limit_h = int(box_h[max_area_idx] * 0.25)
            tl_x = min(tl_x, max_motion_limit_w)
            tl_y = min(tl_y, max_motion_limit_h)
            br_x = max(br_x, w - max_motion_limit_w)
            br_y = max(br_y, h - max_motion_limit_h)

        assert (tl_x < br_x) and (tl_y < br_y)
        top, left, height, width = tl_y, tl_x, (br_y - tl_y), (br_x - tl_x)

        im = np.array(image)
        randomcrop = [CropTransform(left, top, width, height)]

        im, crop_transform = T.apply_augmentations(randomcrop, im)
        crop_box = crop_transform.apply_box(target.gt_boxes.tensor)
        t_h, t_w = im.shape[:2]

        resize = [ResizeTransform(t_h, t_w, h, w)]

        im, resize_transform = T.apply_augmentations(resize, im)
        resize_box = resize_transform.apply_box(crop_box)

        new_target = Instances(target.image_size)
        for field in target.get_fields():
            if field != "gt_boxes":
                new_target.set(field, target.get(field))
            else:
                new_target.set("gt_boxes", Boxes(resize_box))

        # target.set("gt_boxes", Boxes(resize_box))
        return im, new_target

    def __call__(self, video, target=None):

        # Motion augmentation only applies for Siamese Training
        if not isinstance(video, (list, tuple)) or self.motion_limit == 0:
            return video, target

        new_video = []
        new_target = []
        # Only 1 frame go through the motion augmentation,
        # the other unchanged
        idx = random.choice((0, 1))
        for i, (image, image_target) in enumerate(zip(video, target)):
            if i == idx:
                (image, motion_target) = self.crop_motion(image, image_target)
            else:
                motion_target = image_target
            new_video.append(np.array(image))
            new_target.append(motion_target)
        """
        plot_bbox(new_video[0], new_target[0].gt_boxes.tensor)
        plt.imshow()
        """
        return new_video, new_target


class SiamVideoMotionBlurAugment(object):
    """
    SiamVideoMotionBlurAugment Class
    """
    def __init__(self, motion_blur_prob=None):
        self.motion_blur_prob = motion_blur_prob
        if motion_blur_prob is None:
            self.motion_blur_prob = 0.0
        self.motion_blur_func = ImageMotionBlur()

    def __call__(self, video, target):
        # Blur augmentation only applies for Siamese Training
        if not isinstance(video, (list, tuple)) or self.motion_blur_prob == 0.0:
            return video, target

        new_video = []
        new_target = []
        idx = random.choice((0, 1))
        for i, (image, image_target) in enumerate(zip(video, target)):
            if i == idx:
                random_prob = random.uniform(0, 1)
                if random_prob < self.motion_blur_prob:
                    image = self.motion_blur_func(image)
            new_video.append(image)
            new_target.append(image_target)

        return new_video, new_target


class SiamVideoCompressionAugment(object):
    """
    SiamVideoCompressionAugment Class
    """
    def __init__(self, max_compression=None):
        """

        :param max_compression:
        """
        self.max_compression = max_compression
        if max_compression is None:
            self.max_compression = 0.0
        self.compression_func = ImageCompression(self.max_compression)

    def __call__(self, video, target):
        """

        :param video:
        :param target:
        :return:
        """
        # Compression augmentation only applies for Siamese Training
        if not isinstance(video, (list, tuple)) or self.max_compression == 0.0:
            return video, target

        idx = random.choice((0, 1))
        new_video = []
        new_target = []
        for i, (image, image_target) in enumerate(zip(video, target)):
            if i == idx:
                image = self.compression_func(image)
            new_video.append(image)
            new_target.append(image_target)
        return new_video, new_target