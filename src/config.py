from detectron2.config import get_cfg
from detectron2 import model_zoo

import os
import argparse

# Some constants
ROOT_DIR = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
GENERATED_DIR = os.path.join(ROOT_DIR, "generated/")
CHECKPOINT_DIR = os.path.join(ROOT_DIR, "checkpoint/")
VRD_DATASET_PATH = os.path.join(ROOT_DIR, "data/vrd/")
VG200_DATASET_PATH = os.path.join(ROOT_DIR, "data/vg200/")
VRD_IMAGE_OUTPUT_PATH = os.path.join(VRD_DATASET_PATH, "out_images")
VG200_IMAGE_OUTPUT_PATH = os.path.join(VG200_DATASET_PATH, "out_images")
TRIPLES_EMBEDDING_PATH = os.path.join(GENERATED_DIR, "triples_embeddings")


def parse_args():
    """
    Parse Input Arguments for Training
    """
    parser = argparse.ArgumentParser(description="Train a RelTransR network")

    parser.add_argument(
        "--dataset", dest="dataset", required=True, help="Dataset to use"
    )

    # Optimization
    # These options has the highest prioity and can overwrite the values in config file
    # or values set by set_cfgs. `None` means do not overwrite.
    parser.add_argument(
        "--max_iter",
        dest="max_iter",
        help="Explicitly specify to overwrite the value comed from cfg_file.",
        default=None,
        type=int,
    )
    parser.add_argument(
        "--bs",
        dest="batch_size",
        default=1,
        help="Explicitly specify to overwrite the value comed from cfg_file.",
        type=int,
    )
    parser.add_argument(
        "--nw",
        dest="num_workers",
        default=0,
        help="Explicitly specify to overwrite number of workers to load data. Defaults to 0",
        type=int,
    )
    parser.add_argument("--lr", help="Base learning rate.", default=None, type=float)
    parser.add_argument(
        "--lr_decay_gamma", help="Learning rate decay rate.", default=None, type=float
    )

    # Resume trainin
    parser.add_argument(
        "--resume", help="resume to training on a checkpoint", action="store_true"
    )

    return parser.parse_args()


def get_vrd_cfg(args={}, model_conf='faster_rcnn_R_101_FPN_3x.yaml'):
    """
    Setup Configurations for the VRD Dataset
    model:
        - faster_rcnn_R_101_FPN_3x.yaml
        - faster_rcnn_R_50_FPN_3x.yaml
    """
    # detectron 2 config (initialize with default coco config)
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file("COCO-Detection/{}".format(model_conf))
    )

    # Dataset Config
    cfg.DATASETS.TRAIN = ("vrd_train",)
    cfg.DATASETS.TEST = ("vrd_val",)
    cfg.TEST.EVAL_PERIOD = 7560

    # DataLoader Config
    cfg.DATALOADER.NUM_WORKERS = 0

    # Backbone network (initialize from pre-trained model zoo) - will not be used if `resume` training instead
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-Detection/{}".format(model_conf)
    )

    # Solver Learning Rates
    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.SOLVER.MAX_ITER = 75600 if ("max_iter" not in args) else args["max_iter"]

    # Scheduler
    cfg.SOLVER.BASE_LR = 0.0001  # change this to adaptive weights later
    cfg.SOLVER.GAMMA = 0.1
    cfg.SOLVER.STEPS = (53760, 60480)

    # Region of Interests
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512  # number of regions of interests
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 100
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # might have to change to finetune (smaller if there are a lot of small objects in image)
    cfg.MODEL.DEVICE = "cuda"

    # Checkpoint period
    cfg.SOLVER.CHECKPOINT_PERIOD = 1500

    # Output directory
    cfg.OUTPUT_DIR = os.path.join(ROOT_DIR, "checkpoint", "resnet")

    # Custom configurations
    cfg.VRD_RESNETS101_PRETRAINED_WEIGHTS = os.path.join(
        cfg.OUTPUT_DIR, "model_final.pth"
    )

    return cfg
