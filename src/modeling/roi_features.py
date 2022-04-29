from detectron2.modeling import build_backbone
from detectron2.modeling.poolers import ROIPooler

from config import get_vrd_cfg


def get_roi_features(images, box_lists, output_size=(14, 14)):
    """
    Get image features from the backbone network
    Input:
        images: (ImageList.from_tensors) with dimension (B,C,W,H)
        box_lists: A list of N boxes
    """
    cfg = get_vrd_cfg()
    backbone = build_backbone(cfg)
    pooler = ROIPooler(
        output_size,
        pooler_type="ROIAlignV2",
        scales=[1 / 4, 1 / 8, 1 / 16, 1 / 32, 1 / 64],
        sampling_ratio=4,
    )
    feature_maps = backbone(images)
    feature_maps = [feature_maps["p{}".format(i)] for i in range(2, 7)]
    regions_feature = pooler(feature_maps, box_lists)
    print(regions_feature.shape)

    return regions_feature


# Another way to perform inference
# model = build_model(get_vrd_cfg())
# DetectionCheckpointer(model).load("output/model_final.pth")
# evaluator = COCOEvaluator("balloon_val", cfg, False, output_dir="./output/")
# val_loader = build_detection_test_loader(cfg, "balloon_val")
# inference_on_dataset(model, val_loader, evaluator)


# Pytorch forward hook to access intermediate layers
# https://discuss.pytorch.org/t/how-can-l-load-my-best-model-as-a-feature-extractor-evaluator/17254/6

# Relevant Links
# https://stackoverflow.com/questions/62442039/detectron2-extract-region-features-at-a-threshold-for-object-detection
# https://detectron2.readthedocs.io/en/latest/modules/modeling.html#module-detectron2.modeling.poolers
# https://github.com/facebookresearch/detectron2/issues/5
