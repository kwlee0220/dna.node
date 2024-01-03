'''
@Project ：ETRI
@File ：image_augmentation.py
@Author ： Hyunseop Kim
@Date ：22. 11. 22.
'''

import torch
import random
import numpy as np
from PIL import Image

import imgaug.augmenters as iaa

# from ..transform import transform as T
from ..transform import ResizeTransform, CropTransform, Resize
from dna.track.qdtrack.models.structures import Boxes, Instances
import dna.track.qdtrack.models.data.adapters.transform as T


class ImageResize(object):
    """
    ImageResize Class
    """
    def __init__(self, min_size, max_size, size_divisibility):
        """
        init function
        """
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.min_size = min_size
        self.max_size = max_size
        self.size_divisibility = size_divisibility
        # self.resize = T.ResizeShortestEdge(short_edge_length=list(min_size),
        #                                    max_size=max_size, sample_style='choice')

    # modified from torchvision to add support for max size
    def get_size(self, image_size):
        """
        get_size
        """
        h, w = image_size
        size = random.choice(self.min_size)
        max_size = self.max_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(
                    round(max_size * min_original_size / max_original_size))

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        if self.size_divisibility > 0:
            oh = (int(oh / self.size_divisibility) * self.size_divisibility)
            ow = (int(ow / self.size_divisibility) * self.size_divisibility)
        return (oh, ow)

    def __call__(self, image, target=None):
        if not isinstance(image, (np.ndarray)):
            image = np.array(image)
        size = self.get_size(image.shape[:2])
        resize = [Resize(size)]
        image, transform = T.apply_augmentations(resize, np.array(image))
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


class ImageCropResize(object):
    """
    Crop a patch from the image and resize to its original size
    """

    def __init__(self, crop_limit=None, amodal=False):
        self.crop_limit = crop_limit
        self.amodal = amodal

    def __call__(self, image, target):
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

        target.set("gt_boxes", Boxes(resize_box))
        return im, target


class ImageMotionBlur(object):
    """
    Perform motion augmentation to an image
    """

    def __init__(self):
        motion_blur = iaa.MotionBlur(k=10, angle=[-30, 30])
        gaussian_blur = iaa.GaussianBlur(sigma=(0.0, 2.0))

        self.blur_func_pool = [motion_blur, gaussian_blur]

        pass

    def __call__(self, image):
        blur_id = random.choice(list(range(0, len(self.blur_func_pool))))
        blur_func = self.blur_func_pool[blur_id]
        np_image = np.asarray(image)
        blurred_image = blur_func.augment_image(np_image)
        pil_image = Image.fromarray(np.uint8(blurred_image))
        return pil_image


class ImageCompression(object):
    """
    Perform JPEG compression augmentation to an image
    """

    def __init__(self, max_compression):
        self.max_compression = max_compression

    def __call__(self, image):
        ratio = random.uniform(0, 1)
        compression = min(100, int(ratio * self.max_compression))
        np_image = np.asarray(image)
        compressed_image = iaa.arithmetic.compress_jpeg(np_image, compression)
        pil_image = Image.fromarray(np.uint8(compressed_image))
        return pil_image


class ToTensor(object):
    """
    ToTensor Class
    """
    def __call__(self, image, target=None):
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        return image, target


class ToBGR255(object):
    """
    ToBGR255 Class
    """
    def __init__(self, to_bgr255=True):
        self.to_bgr255 = to_bgr255

    def __call__(self, image, target=None):
        if self.to_bgr255:
            image = image[[2, 1, 0]] * 255
        return image, target
