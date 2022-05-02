import os
import cv2
import json
import torch
import random
import copy
import numpy as np
from utils.boxes import boxes_union
from torch.utils.data import Dataset, DataLoader

from torch.utils.data import Dataset, DataLoader
import detectron2.data.transforms as T
import torch.multiprocessing
import matplotlib.pyplot as plt

torch.multiprocessing.set_sharing_strategy("file_system")

from config import VRD_DATASET_PATH

# custom transformation
detectron_transform = T.Resize((800, 800))


class VRDDataset(Dataset):
    def __init__(self, set_type="train", transform=detectron_transform):
        """
        Note:
            transform only applys on bounding boxes. The transformation to the image should be done by detectron2 dataloader.
        """
        annotations_path = os.path.join(
            VRD_DATASET_PATH, "new_annotations_{}.json".format(set_type)
        )
        self.images_dir = os.path.join(VRD_DATASET_PATH, "{}_images".format(set_type))
        self.transform = transform

        with open(annotations_path) as fp:
            raw_annotation = json.load(fp)
        self.annotations = list(raw_annotation.items())

        # check if the data is pre-generated
        roidb_chkpt = os.path.join(VRD_DATASET_PATH, "vrd_roidb_{}.json")
        if os.path.exists(roidb_chkpt):
            with open(roidb_chkpt) as fp:
                self.roidb = json.load(fp)
        else:
            roidb = []
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
                        boxes_union(np.array([subj["bbox"]]), np.array([obj["bbox"]]))[
                            0
                        ]
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

                roidb.append(
                    {
                        # Detectron
                        "file_name": os.path.join(self.images_dir, img_name),
                        "image_id": int(img_name.split(".")[0]),
                        "annotations": list(unique_objects.values()),
                        # Relationships
                        "relationships": {
                            "subj_bboxes": subj_bboxes,
                            "obj_bboxes": obj_bboxes,
                            "union_bboxes": union_bboxes,
                            "subj_classes": subj_classes,
                            "pred_classes": pred_classes,
                            "obj_classes": obj_classes,
                        },
                    }
                )
            self.roidb = roidb

    def __len__(self):
        return len(self.roidb)

    def __getitem__(self, idx):
        item = self.roidb[idx]
        cv2.setNumThreads(0)
        image = cv2.imread(item["file_name"])

        # get transformation
        auginput = T.AugInput(image)
        transform = self.transform(auginput)
        auginput2 = T.AugInput(image)
        transform2 = self.transform(auginput2)
        relationships = item["relationships"]

        # update bboxes
        subj_bboxes = []
        obj_bboxes = []
        union_bboxes = []

        for subj_bbox, obj_bbox in zip(
            relationships["subj_bboxes"], relationships["obj_bboxes"]
        ):
            new_subj_box = transform.apply_box(subj_bbox)
            new_obj_box = transform2.apply_box(obj_bbox)
            new_union_box = boxes_union(
                copy.deepcopy(new_subj_box), copy.deepcopy(new_obj_box)
            )[0]

            subj_bboxes.append(new_subj_box)
            obj_bboxes.append(new_obj_box)
            union_bboxes.append(new_union_box)
        relationships["subj_bboxes"] = subj_bboxes
        relationships["obj_bboxes"] = obj_bboxes
        relationships["union_bboxes"] = union_bboxes
        # add height and width
        height, width = image.shape[:2]
        item["height"] = height
        item["width"] = width

        return item


def get_object_classes(set_name):
    if set_name == "vrd":
        classes = []
        with open(os.path.join(VRD_DATASET_PATH, "objects.json")) as fp:
            classes = json.load(fp)
        return classes
    elif set_name == "vg":
        # to be implemented
        pass
    else:
        raise NotImplementedError


def get_predicate_classes(set_name):
    if set_name == "vrd":
        classes = []
        with open(os.path.join(VRD_DATASET_PATH, "predicates.json")) as fp:
            classes = json.load(fp)
        # add unknown predicate class for missing predicates
        # classes.insert(0, 'unknown')
        classes.append("unrelated")
        return classes
    elif set_name == "vg":
        # to be implemented
        pass
    else:
        raise NotImplementedError


def visualize_bboxes(dataset):
    pred_classes = get_predicate_classes("vrd")
    obj_classes = get_object_classes("vrd")

    for i in random.sample(range(len(dataset)), 60):
        image, cropped_img, anno = dataset[i]
        img = copy.deepcopy(image)
        subj = anno["subject"]
        obj = anno["object"]

        subject_class = obj_classes[subj["category"]]
        predicate_class = pred_classes[anno["predicate"]]
        object_class = obj_classes[obj["category"]]

        img = cv2.rectangle(img, subj["bbox"][0:2], subj["bbox"][2:4], (0, 0, 255), 2)
        img = cv2.rectangle(img, obj["bbox"][0:2], obj["bbox"][2:4], (255, 0, 0), 2)

        cv2.imshow(
            " ".join((subject_class, predicate_class, object_class)), cropped_img
        )
        cv2.waitKey(3000)


def visualize_image_bboxes(image, instances, object_classes=None):
    """
    Inputs:
        image: image tensor
        instances: Instances object from detectron2
    """
    img = copy.deepcopy(image)
    instances_dict = instances[0].get_fields()
    bounding_boxes = [box.tolist() for box in instances_dict["pred_boxes"]]
    labels = instances_dict["pred_classes"]

    for bbox in bounding_boxes:
        img = cv2.rectangle(
            img,
            [int(coord) for coord in bbox[0:2]],
            [int(coord) for coord in bbox[2:4]],
            (255, 255, 255),
            2,
        )
    plt.imshow(img)
    plt.show()


def visualize_bboxes(dataset):
    pred_classes = get_predicate_classes("vrd")
    obj_classes = get_object_classes("vrd")

    for i in random.sample(range(len(dataset)), 60):
        image, cropped_img, anno = dataset[i]
        img = copy.deepcopy(image)
        subj = anno["subject"]
        obj = anno["object"]

        subject_class = obj_classes[subj["category"]]
        predicate_class = pred_classes[anno["predicate"]]
        object_class = obj_classes[obj["category"]]

        img = cv2.rectangle(img, subj["bbox"][0:2], subj["bbox"][2:4], (0, 0, 255), 2)
        img = cv2.rectangle(img, obj["bbox"][0:2], obj["bbox"][2:4], (255, 0, 0), 2)

        cv2.imshow(
            " ".join((subject_class, predicate_class, object_class)), cropped_img
        )
        cv2.waitKey(3000)


if __name__ == "__main__":
    dataset = VRDDataset(set_type="train")
    dataloader = DataLoader(dataset, batch_size=1)

    for item in iter(dataloader):
        print(item)
        break
