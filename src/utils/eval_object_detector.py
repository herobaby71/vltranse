import os
import torch
import numpy as np
import seaborn as sns
import pandas as pd

from config import get_vrd_cfg
from utils.trainer import load_checkpoint
from modeling.vltranse_256 import VLTransE

from detectron2.data import (
    DatasetCatalog,
    DatasetMapper,
    build_detection_train_loader,
    build_detection_test_loader,
)
from config import get_vrd_cfg
import detectron2.data.transforms as T
from utils.dataset import get_object_classes

from collections import defaultdict
from detectron2.evaluation import (
    COCOEvaluator,
    DatasetEvaluator,
    DatasetEvaluators,
    inference_on_dataset,
)


class Counter(DatasetEvaluator):
    def reset(self):
        self.count = 0
        self.gt_count = 0
        self.object_areas = defaultdict(list)
        self.object_classes = get_object_classes()

    def process(self, inputs, outputs):
        for output in outputs:
            self.count += len(output["instances"])

        for inp in inputs:
            self.gt_count += len(inp["instances"])

            annotations = inp["instances"].get_fields()
            for box, cls in zip(annotations["gt_boxes"], annotations["gt_classes"]):
                object_width = box[2] - box[0] + 1
                object_height = box[3] - box[1] + 1
                object_area = object_width * object_height
                self.object_areas["area-{}".format(self.object_classes[cls])].append(
                    object_area
                )

    def evaluate(self):
        # save self.count somewhere, or print it, or return it.
        resulting_dict = {
            "count": self.count,
            "gt_count": self.gt_count,
            "object_average_area": {},
        }

        for key, areas in self.object_areas.items():
            resulting_dict["object_average_area"][key] = sum(areas) / len(areas)

        return resulting_dict


def beatify_detectron2_results(eval_results):
    """
    Beautify the results output by detectron2
    """
    object_areas = eval_results["object_average_area"]
    object_area_ap = {}

    for eval_method, eval_result in eval_results.items():
        if eval_method == "count":
            print("Total Objects Detected:", eval_result)
        elif eval_method == "gt_count":
            print("Total Labeled Objects:", eval_result)
        elif eval_method == "object_average_area":
            continue
        else:
            print("Evaluation results for {}".format(eval_method))

            resulting_string = ""
            for i, (key, res) in enumerate(eval_result.items()):
                resulting_string = "".join(
                    (resulting_string, "|   {:>16}\t->\t{:5.2f}".format(key, res))
                )
                if (i + 1) <= 6:
                    resulting_string = "".join((resulting_string, "   |"))
                if (i + 1) == 6:
                    resulting_string = "".join(
                        (resulting_string, "\nEvaluation results by object category\n")
                    )
                elif (i + 1) > 6:
                    object_cls = key.split("-")[1]
                    area_key = "area-{}".format(object_cls)
                    object_area = object_areas[area_key]

                    resulting_string = "".join(
                        (resulting_string, "( {:5.2f} area )\t|".format(object_area))
                    )
                    object_area_ap[object_cls] = (res, object_area)

                if (i + 1) % 2 == 0:
                    resulting_string = "".join((resulting_string, "\n"))
            print(resulting_string)

    return object_area_ap


def main():
    object_classes = get_object_classes("vrd")

    cfg = get_vrd_cfg()
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6
    cfg.DATASETS.TEST = ("vrd_val",)

    # test dataset
    test_dataset = DatasetCatalog.get("vrd_val")
    test_dataloader = build_detection_test_loader(
        dataset=test_dataset,
        mapper=DatasetMapper(cfg, is_train=True, augmentations=[T.Resize((800, 800))]),
    )

    # train dataset
    train_dataset = DatasetCatalog.get("vrd_train")
    train_dataloader = build_detection_test_loader(
        dataset=train_dataset,
        mapper=DatasetMapper(cfg, is_train=True, augmentations=[T.Resize((800, 800))]),
    )

    # In[12]:

    model = VLTransE(cfg)
    device = torch.device("cuda")
    model.to(device)

    # Run only once
    model.eval()

    chkpoint_path = "../checkpoint/"
    model_name = "vrd2_model_transr_23000.pt"
    chkpoint_full_path = os.path.join(chkpoint_path, model_name)
    _, _, _ = load_checkpoint(model, chkpoint_full_path)

    # ##### Object detector performance for Test Dataset

    eval_results = inference_on_dataset(
        model.detectron,
        test_dataloader,
        DatasetEvaluators(
            [
                COCOEvaluator(
                    "vrd_val", output_dir="../generated/coco_evaluations_val"
                ),
                Counter(),
            ]
        ),
    )

    object_area_ap = beatify_detectron2_results(eval_results)

    test_object_data = [(key, item[0], item[1]) for key, item in object_area_ap.items()]

    area_ap_df = pd.DataFrame(
        pd.DataFrame(list(test_object_data), columns=["Object", "AP", "Area"])
    )

    sns.scatterplot(data=area_ap_df, x="Area", y="AP")

    rank_ap = np.argsort([item[1] for item in test_object_data])
    ranked_object = np.array([item[0] for item in test_object_data])[rank_ap]
    print("Worst Performing Objects:", ranked_object[:20])

    # ##### Object detector performance for Train Dataset

    eval_results = inference_on_dataset(
        model.detectron,
        train_dataloader,
        DatasetEvaluators(
            [
                COCOEvaluator(
                    "vrd_train", output_dir="../generated/coco_evaluations_train/"
                ),
                Counter(),
            ]
        ),
    )

    object_area_ap = beatify_detectron2_results(eval_results)
    train_object_data = [
        (key, item[0], item[1]) for key, item in object_area_ap.items()
    ]

    area_ap_df = pd.DataFrame(
        pd.DataFrame(list(train_object_data), columns=["Object", "AP", "Area"])
    )

    sns.scatterplot(data=area_ap_df, x="Area", y="AP")

    rank_ap = np.argsort([item[1] for item in train_object_data])
    ranked_object = np.array([item[0] for item in train_object_data])[rank_ap]
    print("Worst Performing Objects:", ranked_object[:20])
