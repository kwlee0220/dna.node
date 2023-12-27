from typing import Optional

from pathlib import Path
import logging

import numpy as np
from scipy.stats import multivariate_normal
import torch
import torchvision

from dna import Box
from dna.camera import Image
from dna.detect import Detection
from .types import MetricExtractor


class DeepSORTMetricExtractor(MetricExtractor):
    def __init__(self, wt_path:Path, normalize:bool=False) -> None:
        #loading this encoder is slow, should be done only once.	
        self.encoder = torch.load(wt_path)			
        self.encoder = self.encoder.cuda()
        self.encoder = self.encoder.eval()
        self.normalize = normalize

        self.gaussian_mask = get_gaussian_mask().cuda()
        self.transforms = torchvision.transforms.Compose([ \
                            torchvision.transforms.ToPILImage(),\
                            torchvision.transforms.Resize((128,128)),\
                            torchvision.transforms.ToTensor()])
        
    def distance(self, metric1:np.ndarray, metric2:np.ndarray) -> float:
        if not self.normalize:
            metric1 = metric1 / np.linalg.norm(metric1, axis=1, keepdims=True)
            metric2 = metric2 / np.linalg.norm(metric2, axis=1, keepdims=True)
            
        return 1. - np.dot(metric1, metric2.T)
        # return np.inner(metric1, metric2)
    
    def extract_crops(self, crops:list[Image]) -> np.ndarray:
        processed_crops = [self.transforms(crop) for crop in crops]
        processed_crops = torch.stack(processed_crops)
        processed_crops = processed_crops.cuda()
        processed_crops = self.gaussian_mask * processed_crops

        features = self.encoder.forward_once(processed_crops)
        features = features.detach().cpu().numpy()
        if len(features.shape)==1:
            features = np.expand_dims(features,0)
        if self.normalize:
            features = features / np.linalg.norm(features, axis=1, keepdims=True)
        return features
        
    def extract_boxes(self, image:Image, boxes:list[Box]) -> np.ndarray:
        tlwh_list = [box.tlwh for box in boxes]
        processed_crops = self.pre_process(image, tlwh_list).cuda()
        processed_crops = self.gaussian_mask * processed_crops

        features = self.encoder.forward_once(processed_crops)
        features = features.detach().cpu().numpy()
        if len(features.shape)==1:
            features = np.expand_dims(features,0)
        if self.normalize:
            features = features / np.linalg.norm(features, axis=1, keepdims=True)
        return features

    def extract_dets(self, image:Image, detections:list[Detection]):
        if detections:
            tlwh_list = [det.bbox.tlwh for det in detections]
            processed_crops = self.pre_process(image, tlwh_list).cuda()
            processed_crops = self.gaussian_mask * processed_crops

            features = self.encoder.forward_once(processed_crops)
            features = features.detach().cpu().numpy()
            if len(features.shape)==1:
                features = np.expand_dims(features,0)
            if self.normalize:
                features = features / np.linalg.norm(features, axis=1, keepdims=True)
            return features
        else:
            return []
 
    def pre_process(self, frame, tlwhs):
        crops = []
        for d in tlwhs:
            for i in range(len(d)):
                if d[i] <0:
                    d[i] = 0	

            img_h,img_w,img_ch = frame.shape

            xmin,ymin,w,h = d

            if xmin > img_w:
                xmin = img_w

            if ymin > img_h:
                ymin = img_h

            xmax = xmin + w
            ymax = ymin + h

            ymin = abs(int(ymin))
            ymax = abs(int(ymax))
            xmin = abs(int(xmin))
            xmax = abs(int(xmax))

            try:
                crop = frame[ymin:ymax,xmin:xmax,:]
                crop = self.transforms(crop)
                crops.append(crop)
            except:
                continue

        return torch.stack(crops)

def get_gaussian_mask():
	#128 is image size
	x, y = np.mgrid[0:1.0:128j, 0:1.0:128j]
	xy = np.column_stack([x.flat, y.flat])
	mu = np.array([0.5,0.5])
	sigma = np.array([0.22,0.22])
	covariance = np.diag(sigma**2) 
	z = multivariate_normal.pdf(xy, mean=mu, cov=covariance) 
	z = z.reshape(x.shape) 

	z = z / z.max()
	z  = z.astype(np.float32)

	mask = torch.from_numpy(z)

	return mask

class TestEuclideanMetricExtractor(DeepSORTMetricExtractor):
    def __init__(self, wt_path: Path) -> None:
        super().__init__(wt_path, normalize=True)
        
    def distance(self, metric1:np.ndarray, metric2:np.ndarray) -> float:
        l2 = np.linalg.norm(metric1 - metric2)
        return (2 - l2*l2) / 2