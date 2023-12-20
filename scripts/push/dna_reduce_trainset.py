from __future__ import annotations

from pathlib import Path
import shutil

import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

import dna
from dna.track.dna_tracker import load_feature_extractor, DeepSORTMetricExtractor


import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Generate ReID training data set")
    parser.add_argument("trainset")
    parser.add_argument("output")
    parser.add_argument("--min_k", metavar="count", type=int, default=16, help="minimum cluster count")
    parser.add_argument("--max_k", metavar="count", type=int, default=28, help="maximum cluster count")
    parser.add_argument("--logger", metavar="file path", help="logger configuration file path")
    return parser.parse_known_args()


def do_kmeans(feature_extractor:DeepSORTMetricExtractor, gtrack:Path, min_k:int, max_k:int) -> list[Path]:
    crop_files = [crop_file for crop_file in gtrack.glob('*.png')]
    crops = [cv2.imread(str(crop_file), cv2.IMREAD_COLOR) for crop_file in crop_files]
    features = feature_extractor.extract_crops(crops)
    features = np.asarray(features) / np.linalg.norm(features, axis=1, keepdims=True)
    
    best:KMeans = None
    start_k = min(min_k, len(features))
    stop_k = min(max_k, len(features))
    for k in range(start_k, stop_k+1):
        kmeans = KMeans(n_clusters=k, init='random', n_init=10, random_state=777)
        kmeans.fit(features)
        if not best or kmeans.inertia_ < best.inertia_:
            best = kmeans
            if best.inertia_ < 2.0:
                break
    
    print(f'track={gtrack.name}, k={best.n_clusters}, SSE={best.inertia_:.2f}')
    centroid_crop_files = [None] * best.n_clusters
    for label, center in enumerate(best.cluster_centers_):
        member_idxes = [idx for idx, feat_label in enumerate(best.labels_) if label == feat_label]
        closest_member_idx = np.argmin([np.linalg.norm(features[idx] - center) for idx in member_idxes])
        closest_f_idx = member_idxes[closest_member_idx]
        centroid_crop_files[label] = crop_files[closest_f_idx]
        
    return centroid_crop_files


def main():
    args, _ = parse_args()

    dna.initialize_logger(args.logger)
    assert args.min_k <= args.max_k
    
    feature_extractor = load_feature_extractor()
    
    trainset = Path(args.trainset)
    target_trainset = Path(args.output)
    target_trainset.mkdir(parents=True, exist_ok=True)
    assert target_trainset.exists()
    for gtrack in trainset.iterdir():
        target_gtrack = target_trainset / gtrack.name
        target_gtrack.mkdir(parents=True, exist_ok=True)
        for center in do_kmeans(feature_extractor, gtrack, min_k=args.min_k, max_k=args.max_k):
            target = target_gtrack / center.name
            shutil.copy(center, target)

if __name__ == '__main__':
	main()