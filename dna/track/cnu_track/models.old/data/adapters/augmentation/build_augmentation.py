'''
@Project ：ETRI
@File ：build_augmentation.py
@Author ： Hyunseop Kim
@Date ：22. 11. 22.
'''
from .video_augmentation import SiamVideoResize, \
    SiamVideoColorJitter, SiamVideoCompressionAugment, SiamVideoMotionAugment, \
    SiamVideoMotionBlurAugment, SiamVideoRandomHorizontalFlip, VideoTransformer
from dna.track.qdtrack.models.data.adapters.augmentation.image_augmentation import ToTensor


def build_augmentation(cfg, is_train=True):
    """
    :param cfg: Detectron2 config
    :param is_train: training option
    :return: augmentation functions
    """
    motion_limit = 0.0
    compression_limit = 0.0
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        flip_horizontal_prob = 0.5  # cfg.INPUT.FLIP_PROB_TRAIN
        brightness = cfg.INPUT.BRIGHTNESS
        contrast = cfg.INPUT.CONTRAST
        saturation = cfg.INPUT.SATURATION
        hue = cfg.INPUT.HUE
        motion_limit = cfg.INPUT.MOTION_LIMIT
        compression_limit = cfg.INPUT.COMPRESSION_LIMIT

    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        flip_horizontal_prob = 0.0
        brightness = 0.0
        contrast = 0.0
        saturation = 0.0
        hue = 0.0

    SIZE_DIVISIBILITY = cfg.DATALOADER.SIZE_DIVISIBILITY

    video_color_jitter = SiamVideoColorJitter(
        brightness=brightness,
        contrast=contrast,
        saturation=saturation,
        hue=hue,
    )

    if is_train:
        transform = Compose(
            [
                video_color_jitter,
                SiamVideoCompressionAugment(compression_limit),
                SiamVideoMotionAugment(motion_limit, False),
                SiamVideoResize(min_size, max_size, SIZE_DIVISIBILITY),
                SiamVideoRandomHorizontalFlip(prob=flip_horizontal_prob),
                # PIL image
                VideoTransformer(ToTensor()),
            ]
        )
    else:
        transform = Compose(
            [
                SiamVideoResize(min_size, max_size, SIZE_DIVISIBILITY),
                VideoTransformer(ToTensor()),
            ]
        )
    return transform


class Compose(object):
    """
    Compose Class
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target=None):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string