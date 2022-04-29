#!/usr/bin/env python
# coding: utf-8


import os
import torch
import wandb
import time


from torch.optim.lr_scheduler import StepLR
from utils.register_dataset import register_vrd_dataset
from config import get_vrd_cfg, CHECKPOINT_DIR


from modeling.vltranse_256 import VLTransE


from utils.annotations import get_object_classes, get_predicate_classes

from modeling.word_features import get_triples_features, get_trained_triples_memo

from detectron2.data import (
    DatasetCatalog,
    DatasetMapper,
    build_detection_train_loader,
    build_detection_test_loader,
)
import detectron2.data.transforms as T
from utils.trainer import load_checkpoint


def train_log(loss, lr, it, epoch, loss_subj, loss_pred, loss_obj, loss_transr):
    # Where the magic happens
    wandb.log(
        {
            "lr": lr,
            "epoch": epoch,
            "loss": loss,
            "loss_subj": loss_subj,
            "loss_pred": loss_pred,
            "loss_obj": loss_obj,
            "loss_transr": loss_transr,
        },
        step=it,
    )


def train_model(model_name="vltranse_256"):
    cfg = get_vrd_cfg()
    torch.manual_seed(0)
    dataset_name = "vrd"
    checkpoint_model_name = "vltranse_max_negative_8000.pt"

    # [Only Run once] Register dataset with detectron2 instead of using my own dataloader
    register_vrd_dataset(dataset_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VLTransE(cfg)
    model.to(device)

    # Parallel
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # Data Loader
    train_dataset = DatasetCatalog.get(f"{dataset_name}_train")
    train_dataloader = build_detection_train_loader(
        cfg,
        dataset=train_dataset,
        mapper=DatasetMapper(cfg, is_train=True, augmentations=[T.Resize((800, 800))]),
    )
    iter_dataloader = iter(train_dataloader)

    # How long will the training be?
    n_iters = 12001  # cfg.SOLVER.MAX_ITER
    n_datapoints = len(train_dataset)
    num_epochs = int(n_iters / n_datapoints)
    learning_rate = 0.001
    optimizer = torch.optim.SGD(
        model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True
    )
    scheduler = StepLR(optimizer, step_size=4, gamma=0.1)

    # Checkpoint
    chkpoint_it = n_datapoints  # create a checkpoint every epoch
    initial_it = 0  # checkpoint intial iteration to resume training
    losses = []

    # Load Checkpoint
    load_chkpoint = False
    if load_chkpoint:
        chkpoint_full_path = os.path.join(CHECKPOINT_DIR, checkpoint_model_name)
        it, start_epoch, losses = load_checkpoint(
            model, chkpoint_full_path, optimizer=optimizer
        )
        initial_it = it

    n_datapoints = len(train_dataset)
    interval_cnt = 0

    # WanDB
    project_name = model_name
    log_interval = 20

    wandb.init(project=project_name, entity="herobaby71")
    wandb.config = {
        "seed": 0,
        "learning_rate": learning_rate,
        "gamma": 0.1,
        "momentum": 0.9,
        "epochs": num_epochs,
        "n_iters": n_iters,
        "batch_size": 1,
    }
    wandb.watch(model, log="all", log_freq=20)

    # Losses
    total_loss = 0
    subj_loss = 0
    obj_loss = 0
    pred_loss = 0
    transr_loss = 0

    # start_it
    it = 0
    start_time = time.time()

    for i in range(n_iters):
        # iterator
        try:
            data = next(iter_dataloader)[0]
        except StopIteration:
            print(
                "iterator has reach its end at iteration {}. Initializing a new iterator.".format(
                    str(it)
                )
            )
            iter_dataloader = iter(train_dataloader)
            data = next(iter_dataloader)[0]

        # continue training from the previous checkpoint
        if i < initial_it % n_datapoints:
            continue

        if len(data["relationships"]["subj_bboxes"]) <= 1:
            # image has only one relationship, cannot train
            print("an image has been removed for this batch")
            continue

        # other exclusion due to bad label
        if "1841.jpg" in data["file_name"]:
            print("this image has bad label and has been removed.")
            continue

        optimizer.zero_grad()

        # forward passes
        negative_examples = {}
        negative_examples = model.generate_negative_examples(data, K=100)
        triplet_losses = model(data, negative_examples)

        # compute gradient backward
        final_loss = (
            triplet_losses["obj"]
            + triplet_losses["pred"]
            + triplet_losses["subj"]
            + triplet_losses["transr"]
        )
        final_loss.backward()
        optimizer.step()

        # total loss
        total_loss += final_loss.item()
        subj_loss += triplet_losses["subj"].item()
        pred_loss += triplet_losses["pred"].item()
        obj_loss += triplet_losses["obj"].item()
        transr_loss += triplet_losses["transr"].item()

        interval_cnt += 1
        if it > initial_it and it % log_interval == 0 and it > 0:
            current_loss = total_loss / interval_cnt
            losses.append(current_loss)
            elapsed = time.time() - start_time
            epoch = it / n_datapoints
            print(
                "| it {} | epoch {} | lr {} | ms/batch {:5.2f} | loss {:5.2f}".format(
                    it,
                    int(epoch),
                    scheduler.get_last_lr()[0],
                    elapsed * 1000 / log_interval,
                    current_loss,
                )
            )
            train_log(
                current_loss,
                scheduler.get_last_lr()[0],
                it,
                int(epoch),
                loss_subj=subj_loss / interval_cnt,
                loss_pred=pred_loss / interval_cnt,
                loss_obj=obj_loss / interval_cnt,
                loss_transr=transr_loss / interval_cnt,
            )
            total_loss = 0
            subj_loss = 0
            pred_loss = 0
            obj_loss = 0
            transr_loss = 0
            interval_cnt = 0
            start_time = time.time()

        if it > initial_it and it % chkpoint_it == 0 and it > 0:
            chkpnt = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "it": it,
                "losses": losses,
            }
            torch.save(
                chkpnt, os.path.join(CHECKPOINT_DIR, f"{model_name}_{str(it)}.pt")
            )

        # increment total count
        it = it + 1


train_model()
