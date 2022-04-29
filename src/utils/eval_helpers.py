import collections
import torch
import json
import numpy as np
import detectron2.data.transforms as T

from detectron2.data import DatasetCatalog, DatasetMapper, build_detection_test_loader
from config import get_vrd_cfg
from modeling.word_features import get_trained_triples_memo

eval_config = [
    "triple_dist",  # multiply subj_dist, pred_dist, obj_dist
    "triple_subtract_dist",  # multiply triple_dist with subtract_dist
    "triple_transe_dist",  # multiply triple_dist with transe_dist
    "pred_dist",  # only pred_dist
    "pred_subtract_dist",  # multiply pred_dist with subtract_dist
    "pred_transe_dist",  # multiply pred_dist with transe_dist
]


def get_top_nre_relationships(model, data, nre=50, config="triple_dist"):
    """
    Get the top 50 or 100 relationships for a given input data (image)
    Input:
        data (Dictionary): input dictionary to the model
        nre (Integer): select top n scored relationships
        use_detector_score (Boolean): use the detectron2 detector score
        config (String Selection): see eval_config above
    Return:
        top_n_relationships (Dictionary): top nre relationships based on the `config` scoring metric
    """
    if config not in eval_config:
        print(
            "Error: ensure that the distance metric is one of the following options: ",
            eval_config,
        )

    relationships = data["relationships"]
    (
        predicate_distances,
        subject_distances,
        object_distances,
        predicate_subtract_distances,
        transe_distances,
    ) = model.get_predicate_distances(data, is_rel_eval=True)
    all_possible_relationships = collections.defaultdict(
        list
    )  # {scores: [], subj_classes:[], pred_classes:[], obj_classes:[], subj_bboxes:[], obj_bboxes:[]}
    top_n_relationships = collections.defaultdict(list)  #

    # Iterate through the possible relationships
    for j, (pred_dists, subj_dists, obj_dists) in enumerate(
        zip(predicate_distances, subject_distances, object_distances)
    ):

        # labels for subject and object
        subj_class = relationships["subj_classes"][j]
        obj_class = relationships["obj_classes"][j]
        subj_bbox = relationships["subj_bboxes"][j]
        obj_bbox = relationships["obj_bboxes"][j]
        subj_detection_score = relationships["subj_scores"][j]
        obj_detection_score = relationships["obj_scores"][j]

        # multiply pred, subj, obj
        # Compute scores for the 71 possible predicate and compute the score
        scoring_distances = []
        if config == "triple_dist":
            scoring_distances = [
                item_pred.cpu() * item_subj.cpu() * item_obj.cpu()
                for item_pred, item_subj, item_obj in zip(
                    pred_dists, subj_dists, obj_dists
                )
            ]
        elif config == "pred_dist":
            scoring_distances = [item_pred.cpu() for item_pred in pred_dists]
        elif config == "pred_subtract_dist":
            scoring_distances = [
                item_pred.cpu() * item_subtract.cpu()
                for item_pred, item_subtract in zip(
                    pred_dists, predicate_subtract_distances[j]
                )
            ]
        elif config == "pred_transe_dist":
            scoring_distances = [
                item_pred.cpu() * item_transe.cpu()
                for item_pred, item_transe in zip(pred_dists, transe_distances[j])
            ]
        elif config == "triple_subtract_dist":
            scoring_distances = [
                item_pred.cpu() * item_subj.cpu() * item_obj.cpu() * item_subtract.cpu()
                for item_pred, item_subj, item_obj, item_subtract in zip(
                    pred_dists, subj_dists, obj_dists, predicate_subtract_distances[j]
                )
            ]
        elif config == "triple_transe_dist":
            scoring_distances = [
                item_pred.cpu() * item_subj.cpu() * item_obj.cpu() * item_transe.cpu()
                for item_pred, item_subj, item_obj, item_transe in zip(
                    pred_dists, subj_dists, obj_dists, transe_distances[j]
                )
            ]

        # Adding to all_possible_relationships to rank
        for ind, distance_score in enumerate(scoring_distances):
            all_possible_relationships["subj_classes"].append(subj_class.cpu())
            all_possible_relationships["pred_classes"].append(ind)
            all_possible_relationships["obj_classes"].append(obj_class.cpu())
            all_possible_relationships["subj_bboxes"].append(subj_bbox)
            all_possible_relationships["obj_bboxes"].append(obj_bbox)
            all_possible_relationships["distance_scores"].append(distance_score)
            all_possible_relationships["subj_detection_scores"].append(
                subj_detection_score.detach().cpu().numpy()
            )
            all_possible_relationships["obj_detection_scores"].append(
                obj_detection_score.detach().cpu().numpy()
            )

    # rank to get the top 50, 100 relationships
    sorted_by_distance = np.argsort(all_possible_relationships["distance_scores"])

    for key, val in all_possible_relationships.items():
        sorted_val = np.array(val)[sorted_by_distance]

        top_n_relationships[key] = sorted_val
        if nre is not None and len(val) > nre:
            top_n_relationships[key] = sorted_val[:nre]

    return top_n_relationships


def eval_per_image(
    gt_relationships, pred_relationships, gt_thr=0.5, trained_triples=None, model=None
):
    """
    Iterate through the ground truth relationship of the image and check
        whether they are in the top 50 or 100, while ensuring that the IoU threshhold is above 50%

    Input:
        gt_relationships: ground truth relationships
        pred_relationships: top @n predicted relationships with bounding boxes, etc.
        gr_thr: IoU threshold
        trained_triples: if specified, perform zeroshot evaluation
    Return:
        tp: number of true positives
        n_gt_labels: number of ground truth labels to compute recall
    """
    n_pred_labels = len(pred_relationships["subj_bboxes"])
    n_gt_labels = len(gt_relationships["subj_bboxes"])
    n_zeroshot_gt_labels_set = set()  # only used if trained_triples is specified
    visited = set()  # track if ground truth has be visited

    tp = np.zeros((n_pred_labels, 1))
    fp = np.zeros((n_pred_labels, 1))
    for j in range(n_pred_labels):
        pred_triple_cls = np.array(
            (
                pred_relationships["subj_classes"][j],
                pred_relationships["pred_classes"][j],
                pred_relationships["obj_classes"][j],
            )
        )
        obj_bboxes = pred_relationships["obj_bboxes"][j][0]
        subj_bboxes = pred_relationships["subj_bboxes"][j][0]

        max_iou = gt_thr  # track best overlap
        kmax = -1  # track visited

        for k in range(n_gt_labels):
            gt_triple_cls = np.array(
                (
                    gt_relationships["subj_classes"][k],
                    gt_relationships["pred_classes"][k],
                    gt_relationships["obj_classes"][k],
                )
            )

            if trained_triples is not None:
                gt_subj_cls = model.object_classes[gt_relationships["subj_classes"][k]]
                gt_pred_cls = model.predicate_classes[
                    gt_relationships["pred_classes"][k]
                ]
                gt_obj_cls = model.object_classes[gt_relationships["obj_classes"][k]]
                gt_triple_cls_label = "-".join((gt_subj_cls, gt_pred_cls, gt_obj_cls))

                # if the label is in the training dataset, ignore
                if gt_triple_cls_label in trained_triples:
                    continue

                n_zeroshot_gt_labels_set.add(k)

            # Verify prediction labels match ground truth labels
            if np.linalg.norm(gt_triple_cls - pred_triple_cls) != 0:
                continue
            # Verify that the ground truth labels has not been visited before
            if k in visited:
                continue

            # Check IoU to make sure that the predicted bbox > threshold of 50%
            gt_subj_bboxes = gt_relationships["subj_bboxes"][k][0]
            gt_obj_bboxes = gt_relationships["obj_bboxes"][k][0]

            # Intersection between predicted bbox and gt_bbox
            subj_intersection_bbox = np.array(
                [
                    max(subj_bboxes[0], gt_subj_bboxes[0]),
                    max(subj_bboxes[1], gt_subj_bboxes[1]),
                    min(subj_bboxes[2], gt_subj_bboxes[2]),
                    min(subj_bboxes[3], gt_subj_bboxes[3]),
                ]
            )

            obj_intersection_bbox = np.array(
                [
                    max(obj_bboxes[0], gt_obj_bboxes[0]),
                    max(obj_bboxes[1], gt_obj_bboxes[1]),
                    min(obj_bboxes[2], gt_obj_bboxes[2]),
                    min(obj_bboxes[3], gt_obj_bboxes[3]),
                ]
            )

            subj_intersection_bbox_width = (
                subj_intersection_bbox[2] - subj_intersection_bbox[0] + 1
            )
            subj_intersection_bbox_height = (
                subj_intersection_bbox[3] - subj_intersection_bbox[1] + 1
            )

            obj_intersection_bbox_width = (
                obj_intersection_bbox[2] - obj_intersection_bbox[0] + 1
            )
            obj_intersection_bbox_height = (
                obj_intersection_bbox[3] - obj_intersection_bbox[1] + 1
            )

            # Check overlapping
            if (
                subj_intersection_bbox_width > 0
                and subj_intersection_bbox_height > 0
                and obj_intersection_bbox_width > 0
                and obj_intersection_bbox_height > 0
            ):

                # [subject] compute overlap as area of intersection / area of union
                subj_intersection_area = (
                    subj_intersection_bbox_width * subj_intersection_bbox_height
                )
                subj_union_area = (
                    (subj_bboxes[2] - subj_bboxes[0] + 1)
                    * (subj_bboxes[3] - subj_bboxes[1] + 1)
                    + (gt_subj_bboxes[2] - gt_subj_bboxes[0] + 1)
                    * (gt_subj_bboxes[3] - gt_subj_bboxes[1] + 1)
                    - subj_intersection_area
                )
                subj_iou = subj_intersection_area / subj_union_area

                # [object] compute overlap as area of intersection / area of union
                obj_intersection_area = (
                    obj_intersection_bbox_width * obj_intersection_bbox_height
                )
                obj_union_area = (
                    (obj_bboxes[2] - obj_bboxes[0] + 1)
                    * (obj_bboxes[3] - obj_bboxes[1] + 1)
                    + (gt_obj_bboxes[2] - gt_obj_bboxes[0] + 1)
                    * (gt_obj_bboxes[3] - gt_obj_bboxes[1] + 1)
                    - obj_intersection_area
                )
                obj_iou = obj_intersection_area / obj_union_area

                # only need to evaluate the minimum overlap ratio to test against the threshold
                min_iou = min(subj_iou, obj_iou)

                # makes sure that this object is detected according
                # to its individual threshold
                if min_iou >= max_iou:
                    max_iou = min_iou
                    kmax = k

        if kmax > -1:
            visited.add(kmax)
            tp[j] = 1
        else:
            fp[j] = 1

    if trained_triples is not None:
        return tp.sum(), len(n_zeroshot_gt_labels_set)

    return tp.sum(), n_gt_labels


def eval_dataset(dataloader, model, nre=50, config="triple_dist", trained_triples=None):
    total_true_positive = 0
    total_relationships = 0

    cumulative_recall = []
    recall = 0

    n_examples = len(dataloader.dataset)
    test_data_iter = iter(dataloader)
    for i in range(n_examples):
        data = next(test_data_iter)[0]
        gt_relationships = data["relationships"].copy()
        relationships = None
        top_predicted_relationships = {}

        if len(gt_relationships["subj_bboxes"]) == 0:
            # no relationship annotations for the given image
            continue

        with torch.no_grad():
            # get predicted relationships from the object detector
            relationships = model.get_predicted_relationships(data)
            data["relationships"] = relationships

        if len(relationships["subj_bboxes"]) == 0:
            # no relationship annotations for the given image
            continue

        if i % 20 == 0 and i > 0:
            recall = total_true_positive / total_relationships * 100
            cumulative_recall.append(recall)
            print("|----------------------Iter {}------------------------|".format(i))
            print(
                "| TOP {} |               Recall {:5.2f}               |".format(
                    nre, recall
                )
            )

        # Get the top
        top_predicted_relationships = get_top_nre_relationships(
            model, data, nre, config=config
        )
        image_true_positive, image_relationships = eval_per_image(
            gt_relationships, top_predicted_relationships, 0.5, trained_triples, model
        )

        total_true_positive += image_true_positive
        total_relationships += image_relationships

    recall = total_true_positive / total_relationships * 100
    print("|----------------------Iter {}------------------------|".format(i))
    print("| TOP {} |               Recall {:5.2f}               |".format(nre, recall))
    cumulative_recall.append(recall)

    return cumulative_recall
