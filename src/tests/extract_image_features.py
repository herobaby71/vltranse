import cv2
import sys
sys.path.append("..")

from detectron2.data import (
    DatasetCatalog, DatasetMapper,
    build_detection_train_loader, build_detection_test_loader
)
import detectron2.data.transforms as T
from utils.register_dataset import register_vrd_dataset
from modelling.roi_features import get_roi_features
from config import get_vrd_cfg, VRD_DATASET_PATH
import torch

data_path = VRD_DATASET_PATH

def test_extract_feature_from_images():
    cfg = get_vrd_cfg()
    register_vrd_dataset('vrd')
    dataloader = build_detection_train_loader(cfg,
        mapper=DatasetMapper(cfg, is_train=True, augmentations=[
            T.Resize((800, 800))
        ])
    )
    dataloader_test = build_detection_test_loader(DatasetCatalog.get('vrd_val'), mapper=DatasetMapper(cfg, is_train=False))
    train_features, train_labels = next(iter(dataloader))
    # test_features = next(iter(dataloader_test))[0]
    images = torch.unsqueeze(train_features['image'], axis=0)
    boxes = train_features['instances'].get_fields()['gt_boxes']

    return get_roi_features(images.float(), box_lists=[boxes], output_size=(7, 7))