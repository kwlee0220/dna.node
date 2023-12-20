'''
@Project ：ETRI
@File ：predictor.py
@Author ： Hyunseop Kim
@Date ：22. 11. 22.
'''

import torch
from ..structures import Boxes, BoxMode, Instances
from ..modeling.rcnn import build_models
from ..data.adapters.augmentation.build_augmentation import build_augmentation


class CustomPredictor:
    def __init__(self, cfg):
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.model = build_models(self.cfg)
        self.model.eval()
        checkpointer = torch.load(self.cfg.MODEL.WEIGHTS)
        self.model.load_state_dict(checkpointer)
        self.model.to(cfg.MODEL.DEVICE)
        self.aug = build_augmentation(cfg, is_train=False)
        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format

    def __call__(self, original_images, bounding_boxes=None):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).

        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
        """
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.
            inputs = []
            if self.input_format == "RGB":
                original_images = original_images[:, :, ::-1]

            height, width = original_images.shape[:2]
            if bounding_boxes != None:
                instance = Instances((height, width))
                bbox = torch.tensor(bounding_boxes).to(torch.float)
                labels = torch.Tensor([0 for _ in range(len(bbox))]).to(torch.long)
                score = torch.Tensor([0.95 for _ in range(len(bbox))]).to(torch.float)

                bbox = BoxMode.convert(bbox, BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)
                boxes = Boxes(bbox)

                instance.set("gt_boxes", boxes)
                instance.set("gt_classes", labels)
                instance.set("scores", score)
            else:
                instance = Instances((height, width))
                bbox = torch.Tensor([]).to(torch.float)
                labels = torch.Tensor([]).to(torch.long)
                ids = torch.Tensor([]).to(torch.long)
                score = torch.Tensor([]).to(torch.float)
                #torch.Tensor([i['scores'] for i in ground_truth]).to(torch.float)

                boxes = Boxes(bbox)
                instance.set("gt_boxes", boxes)
                instance.set("gt_classes", labels)
                instance.set("scores", score)

            image, instances = self.aug(original_images, instance)
            inputs = [{"image": image, "height": height, "width": width, "instances": instances}]
            predictions = self.model(inputs)
            return predictions
