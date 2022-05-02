import torch
import copy
import os

from detectron2.engine import DefaultTrainer
from detectron2.data import build_detection_train_loader, build_detection_test_loader
from detectron2.data import detection_utils as utils
from detectron2.data import DatasetCatalog, DatasetMapper
from detectron2.evaluation import COCOEvaluator
import detectron2.data.transforms as T


def load_checkpoint(model, chkpoint_path, optimizer=None):
    chkpoint = torch.load(chkpoint_path)

    # incompatibility fixes
    if (
        "detectron.proposal_generator.anchor_generator.cell_anchors.0"
        in chkpoint["model"]
    ):
        del chkpoint["model"][
            "detectron.proposal_generator.anchor_generator.cell_anchors.0"
        ]
        del chkpoint["model"][
            "detectron.proposal_generator.anchor_generator.cell_anchors.1"
        ]
        del chkpoint["model"][
            "detectron.proposal_generator.anchor_generator.cell_anchors.2"
        ]
        del chkpoint["model"][
            "detectron.proposal_generator.anchor_generator.cell_anchors.3"
        ]
        del chkpoint["model"][
            "detectron.proposal_generator.anchor_generator.cell_anchors.4"
        ]

    model.load_state_dict(chkpoint["model"])
    if optimizer is not None:
        optimizer.load_state_dict(chkpoint["optimizer"])
    return chkpoint["it"], chkpoint["epoch"], chkpoint["losses"]


def custom_mapper(dataset_dict):
    dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
    image = utils.read_image(dataset_dict["file_name"], format="BGR")
    transform_list = [
        T.Resize((800, 800)),
        T.RandomFlip(prob=0.4, horizontal=False, vertical=True),
    ]
    image, transforms = T.apply_transform_gens(transform_list, image)
    dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))
    annos = [
        utils.transform_instance_annotations(obj, transforms, image.shape[:2])
        for obj in dataset_dict.pop("annotations")
        if obj.get("iscrowd", 0) == 0
    ]
    instances = utils.annotations_to_instances(annos, image.shape[:2])
    dataset_dict["instances"] = utils.filter_empty_instances(instances)
    return dataset_dict


class CustomTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=custom_mapper)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name="vrd_val"):
        # build_detection_test_loader(DatasetCatalog.get('vrd_val'), mapper=DatasetMapper(cfg, is_train=False))
        return build_detection_test_loader(
            cfg,
            dataset_name,
            mapper=DatasetMapper(
                cfg, is_train=False, augmentations=[T.Resize((800, 800))]
            ),
        )

    @classmethod
    def build_evaluator(cls, cfg, dataset_name="vrd_val", output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)
