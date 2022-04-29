import os
import json
import cv2
import copy
import torch
import numpy as np
import detectron2.data.transforms as T
import collections

from torch.utils.data import Dataset
from utils.boxes import boxes_union
from src.config import VRD_DATASET_PATH

torch.multiprocessing.set_sharing_strategy("file_system")


class RelDB(Dataset):
    def __init__(self, set_type="train", set_name="vrd", transform=None):
        """
        Get Union bounding boxes, objects' regions, classes, and interactions between them
        """
        dataset_path = VRD_DATASET_PATH if (set_name == "vrd") else ""
        annotations_path = os.path.join(
            dataset_path, f"new_annotations_{set_type}.json"
        )
        self.images_dir = os.path.join(dataset_path, f"{set_type}_images")
        self.transform = transform

        with open(annotations_path) as fp:
            raw_annotation = json.load(fp)
        self.annotations = list(raw_annotation.items())

        # check if the reldb data is pre-generated
        self.reldb = []

        for img_name, annotations in raw_annotation.items():
            subj_bboxes = []
            obj_bboxes = []
            union_bboxes = []
            unique_objects = {}
            subj_classes = []
            obj_classes = []
            pred_classes = []
            for anno in annotations:
                subj = anno["subject"]
                obj = anno["object"]

                subj["bbox"] = [
                    subj["bbox"][2],
                    subj["bbox"][0],
                    subj["bbox"][3],
                    subj["bbox"][1],
                ]  # XYXY
                obj["bbox"] = [
                    obj["bbox"][2],
                    obj["bbox"][0],
                    obj["bbox"][3],
                    obj["bbox"][1],
                ]

                union_bboxes.append(
                    boxes_union(np.array([subj["bbox"]]), np.array([obj["bbox"]]))[0]
                )
                subj_bboxes.append(subj["bbox"])
                obj_bboxes.append(obj["bbox"])
                unique_objects[(tuple(subj["bbox"]))] = {
                    "bbox": subj["bbox"],
                    "bbox_mode": 0,  # BoxMode.XYXY_ABS
                    "category_id": subj["category"],
                }
                unique_objects[(tuple(obj["bbox"]))] = {
                    "bbox": obj["bbox"],
                    "bbox_mode": 0,
                    "category_id": obj["category"],
                }
                subj_classes.append(subj["category"])
                obj_classes.append(obj["category"])
                pred_classes.append(anno["predicate"])

                self.reldb.append(
                    {
                        # Detectron
                        "file_name": os.path.join(self.images_dir, img_name),
                        "image_id": int(img_name.split(".")[0]),
                        "annotations": list(unique_objects.values()),
                        "subj_bbox": subj["bbox"],
                        "union_bbox": boxes_union(
                            np.array([subj["bbox"]]), np.array([obj["bbox"]])
                        )[0],
                        "obj_bbox": obj["bbox"],
                        "subj_class": subj["category"],
                        "pred_class": anno["predicate"],
                        "obj_class": obj["category"],
                    }
                )

        # check if the relationship memo for index is pre-generated
        self.reldb_memo = collections.defaultdict(list)

        for i, rel in enumerate(self.reldb):
            self.reldb_memo[(rel["subj_class"], rel["pred_class"])].append(i)

    def get_memo(self, subj_class, pred_class):
        return self.reldb_memo[(subj_class, pred_class)]

    def __len__(self):
        return len(self.reldb)

    def __getitem__(self, idx):
        item = self.reldb[idx]
        cv2.setNumThreads(0)
        image = cv2.imread(item["file_name"])[..., ::-1]
        original_height, original_width = image.shape[:2]
        print("original (height, width): ", (original_height, original_width))

        # cropped image
        cropped_image = image[
            item["union_bbox"][1] : item["union_bbox"][3],
            item["union_bbox"][0] : item["union_bbox"][2],
        ]
        height, width = cropped_image.shape[:2]
        print("(height, width): ", (height, width))

        # shift the bboxes relative to the cropped image
        item["subj_bbox"] = [
            item["subj_bbox"][0] - item["union_bbox"][0],
            item["subj_bbox"][1] - item["union_bbox"][1],
            item["subj_bbox"][2] - item["union_bbox"][0],
            item["subj_bbox"][3] - item["union_bbox"][1],
        ]
        print("transformed_subj_bbox:", item["subj_bbox"])
        item["obj_bbox"] = [
            item["obj_bbox"][0] - item["union_bbox"][0],
            item["obj_bbox"][1] - item["union_bbox"][1],
            item["obj_bbox"][2] - item["union_bbox"][0],
            item["obj_bbox"][3] - item["union_bbox"][1],
        ]

        # get transformation
        if self.transform:
            auginput = T.AugInput(cropped_image)
            transform = self.transform(auginput)

            # update bboxes (with transformation)
            new_subj_box = transform.apply_box(item["subj_bbox"])
            new_obj_box = transform.apply_box(item["obj_bbox"])
            new_union_box = boxes_union(
                copy.deepcopy(new_subj_box), copy.deepcopy(new_obj_box)
            )[0]

            item["subj_bbox"] = new_subj_box
            item["obj_bbox"] = new_obj_box
            item["union_bbox"] = new_union_box

        # add height and width
        item["height"] = height
        item["width"] = width
        item["image"] = cropped_image

        return item
