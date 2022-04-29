#!/usr/bin/env python
# coding: utf-8


import os
import json
import torch
import wandb
import time


from torch.optim.lr_scheduler import StepLR
from utils.register_dataset import register_vrd_dataset
from config import GENERATED_DIR, get_vrd_cfg, CHECKPOINT_DIR


from modeling.vltranse_256 import VLTransE
from modeling.word_features import get_triples_features, get_trained_triples_memo

from utils.annotations import get_object_classes, get_predicate_classes
from utils.trainer import load_checkpoint
from utils.eval_helpers import eval_dataset, eval_config

from detectron2.data import (
    DatasetCatalog,
    DatasetMapper,
    build_detection_train_loader,
    build_detection_test_loader,
)
import detectron2.data.transforms as T


def load_model(cfg, checkpoint_model_name="vltranse_256_12000.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VLTransE(cfg)
    model.to(device)

    chkpoint_full_path = os.path.join(CHECKPOINT_DIR, checkpoint_model_name)
    _, _, _ = load_checkpoint(model, chkpoint_full_path)

    return model


def test_model(checkpoint_model_name="vltranse_256_12000.pt", dataset_name="vrd_val"):
    cfg = get_vrd_cfg()
    cfg.DATASETS.TEST = ("dataset_name",)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.4

    # [Only Run once] Register dataset with detectron2 instead of using my own dataloader
    register_vrd_dataset(dataset_name.split("_")[0])

    # Load model
    model = load_model(cfg, checkpoint_model_name)
    # Set model to evaluation mode
    model.eval()

    # Parallel
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # For zeroshot
    trained_triples = get_trained_triples_memo()

    # Dataset and DataLoader
    # test_dataset = DatasetCatalog.get(dataset_name)
    # test_dataloader = build_detection_test_loader(
    #     dataset=test_dataset,
    #     mapper=DatasetMapper(cfg, is_train=True, augmentations=[T.Resize((800, 800))]),
    # )

    # Iterate and Test
    recall_results = {}
    i = 0
    for is_zeroshot in [False, True]:
        for nre in [50, 100]:
            for conf in eval_config:
                if i == 0:
                    i += 1
                    continue
                test_dataset = DatasetCatalog.get(dataset_name)
                test_dataloader = build_detection_test_loader(
                    dataset=test_dataset,
                    mapper=DatasetMapper(
                        cfg, is_train=True, augmentations=[T.Resize((800, 800))]
                    ),
                )

                if is_zeroshot:
                    print(("is_zeroshot", nre, conf))
                    recall_results[("is_zeroshot", nre, conf)] = eval_dataset(
                        test_dataloader,
                        model,
                        nre=nre,
                        config=conf,
                        trained_triples=trained_triples,
                    )
                else:
                    print(("not_zeroshot", nre, conf))
                    recall_results[("not_zeroshot", nre, conf)] = eval_dataset(
                        test_dataloader,
                        model,
                        nre=nre,
                        config=conf,
                        trained_triples=None,
                    )

                test_dataloader = None
                del test_dataloader
                del test_dataset
                del model

                model = load_model(cfg, checkpoint_model_name)
                # Set model to evaluation mode
                model.eval()

    recall_results_path = f"{GENERATED_DIR}/{checkpoint_model_name.split('.')[0]}.json"

    with open(recall_results_path, "w") as file:
        file.write(json.dumps(recall_results))


test_model()
