# Improving the model's performance is hard. It is more than changing the network's parameters and configurations
# Therefore, neccessary steps are required to efficiently iterate through the improvement process
# This requires a fully implemented pipeline for testing the model and automate the learning process
# => Steps to improve efficiency and start writing:
#    1/ Ensure the reliability of the evalutation pipeline
#    2/ Improve the model's extensibility and testability
# => Steps to improve the model
#    1/ Improved negative sampling
#    2/ Better objective function
#    3/ Experimenting with the detectors
#    4/ Changing Convolutional-based embeddings to Transformer-based embeddings
#    5/ Contrast the entire concatenated embeddings
#    6/ Allow the word features to be changed
# => Aim: Prove that bigger is not always better, a clear outlines of each experiment
#              with results and analysis of results
#
# Linear transformation and TransR <- is it impossible to create a causual if then relation
# MY CONCLUSION IS THAT THE CURRENT ARCHITECTURE IS IN CAPPABLE OF ACHIEVING THE TASKS WITHOUT INTERACTIONs
#    BETWEEN THE EMBEDDINGS (Will need improve the model steps 4 or 5)


# This work is fundamentally different from Ji Zhang et al.
# I am not contrasting visual embeddings with word embeddings.
# I am contrasting visual relationships embeddings with visual relationships embeddings
# If I contrast obe roi embeddings against all the objects that it is not, it means that the feature is not

# The aim has changed drastically. It is no logner to improve on VRD results.
# The aim is now to make contrastive learning easier to control, no mashing and contrasting.
# Contrast individual embeddings (separability), while maintaining the benefits that contrastive learning gives.

# Essential Issues that need to be resolved:
#    1/ I treat each image as it own space, meaning, the ucrrent technique only contrast visual relationships
#           against other visual relationships within the image. => Lookl global, contrast against visual
#           relationships in other images as well.
#    2/ Negative Sampling is currently only get the first unrelatied instance within the image, and only looking
#           for plausible relationships within the image. Negative Sampling is not considering the scenario where
#           subjects and objects can be similar, but visual relationship is different.
#    3/ Scale (Implement the model in Caffe2)


import os
import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ModuleDict

from utils.annotations import get_object_classes, get_predicate_classes
from utils.boxes import boxes_union

from modeling.roi_features import get_roi_features
from detectron2.modeling import build_model
from detectron2.modeling import build_backbone
from detectron2.modeling.poolers import ROIPooler
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.structures.boxes import Boxes
import random
from collections import defaultdict


class TripletSoftmaxLoss(nn.Module):
    """
    Use softmax with cross entropy instead of argmax for loss function.
    """

    def __init__(self):
        super().__init__()

        self.distance_function = lambda x, y: 1.0 - F.cosine_similarity(x, y)

    def forward(self, mode, fc_features, neg_fc_features_list):
        """
        Compute the Triple Softmax Loss
        Input:
            fc_features: Dictionaries for visual and textual embeddings for different ROIs in the image
            neg_fc_features_list: a list of fc_features of negative examples. Each positive example has
                multiple associated negative examples.
        """

        # number of relationships
        N = fc_features["textual"]["subj"].shape[0]
        
        dist_similarity = self.distance_function()


class VLTransE(nn.Module):
    def __init__(self, cfg, pooling_size=(7, 7), training=True):
        super().__init__()
        self.cfg = cfg

        # Object and Predicate Classes
        self.object_classes = get_object_classes("vrd")
        self.predicate_classes = get_predicate_classes("vrd")

        # Embeddings dimensions
        self.visual_feature_dim = 256 * pooling_size[0] * pooling_size[1]
        self.visual_hidden_dim = 128 * pooling_size[0] * pooling_size[1]
        self.word_feature_dim = 3072
        self.trans_feature_dim = 1024
        self.emb_feature_dim = 256

        # Spatial Module
        self.spatial_feature_dim = 22
        self.spatial_hidden_dim = 64
        self.fc_spatial = torch.nn.Sequential(
            torch.nn.Linear(self.spatial_feature_dim, self.spatial_hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.spatial_hidden_dim, self.spatial_hidden_dim),
        )

        # Visual Modal
        self.detectron = build_model(cfg)
        if training:
            self._load_detectron_chkpoints(cfg)

        # Seperate for predicate
        self.backbone = copy.deepcopy(self.detectron.backbone)
        self.pooler = copy.deepcopy(self.detectron.roi_heads.box_pooler)
        self._freeze_parameters(cfg)

        # Language Modal
        self.bert_model = BertModel.from_pretrained(
            "bert-base-uncased",
            # Whether the model returns all hidden-states.
            output_hidden_states=True,
        )
        self.bert_model.to("cuda")
        self.bert_model.eval()
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        # Pre-trained token embeddings (static without changes for now)
        self.triples_embeddings_path = "../generated/triples_embeddings.pt"
        self.token_embeddings = self._load_words_chkpoints(cfg)

        # Fully connect language
        self.fc_word = ModuleDict(
            {
                "subj": torch.nn.Sequential(
                    torch.nn.Linear(self.word_feature_dim, self.trans_feature_dim),
                    # torch.nn.BatchNorm1d(self.trans_feature_dim),
                    torch.nn.LeakyReLU(0.1),
                    torch.nn.Linear(self.trans_feature_dim, self.emb_feature_dim),
                ),
                "pred": torch.nn.Sequential(
                    torch.nn.Linear(self.word_feature_dim, self.trans_feature_dim),
                    # torch.nn.BatchNorm1d(self.trans_feature_dim),
                    torch.nn.LeakyReLU(0.1),
                    torch.nn.Linear(self.trans_feature_dim, self.emb_feature_dim),
                ),
                "obj": torch.nn.Sequential(
                    torch.nn.Linear(self.word_feature_dim, self.trans_feature_dim),
                    # torch.nn.BatchNorm1d(self.trans_feature_dim),
                    torch.nn.LeakyReLU(0.1),
                    torch.nn.Linear(self.trans_feature_dim, self.emb_feature_dim),
                ),
            }
        )

        # Fully connect roi
        self.fc_rois = ModuleDict(
            {
                "subj": torch.nn.Sequential(
                    torch.nn.Linear(self.visual_feature_dim, self.visual_hidden_dim),
                    # torch.nn.BatchNorm1d(self.visual_hidden_dim),
                    torch.nn.LeakyReLU(0.1),
                    torch.nn.Linear(self.visual_hidden_dim, self.trans_feature_dim),
                    # torch.nn.BatchNorm1d(self.trans_feature_dim),
                    torch.nn.LeakyReLU(0.1),
                    torch.nn.Linear(self.trans_feature_dim, self.trans_feature_dim),
                ),
                "pred": torch.nn.Sequential(
                    torch.nn.Linear(
                        self.visual_feature_dim + self.spatial_hidden_dim,
                        self.visual_hidden_dim,
                    ),
                    # torch.nn.BatchNorm1d(self.visual_hidden_dim),
                    torch.nn.LeakyReLU(0.1),
                    torch.nn.Linear(self.visual_hidden_dim, self.trans_feature_dim),
                    # torch.nn.BatchNorm1d(self.trans_feature_dim),
                    torch.nn.LeakyReLU(0.1),
                    torch.nn.Linear(self.trans_feature_dim, self.trans_feature_dim),
                ),
                "obj": torch.nn.Sequential(
                    torch.nn.Linear(self.visual_feature_dim, self.visual_hidden_dim),
                    # torch.nn.BatchNorm1d(self.visual_hidden_dim),
                    torch.nn.LeakyReLU(0.1),
                    torch.nn.Linear(self.visual_hidden_dim, self.trans_feature_dim),
                    # torch.nn.BatchNorm1d(self.trans_feature_dim),
                    torch.nn.LeakyReLU(0.1),
                    torch.nn.Linear(self.trans_feature_dim, self.trans_feature_dim),
                ),
            }
        )

        self.fc_rois2 = ModuleDict(
            {
                "subj": torch.nn.Sequential(
                    torch.nn.Linear(self.trans_feature_dim, self.emb_feature_dim),
                    # torch.nn.BatchNorm1d(self.emb_feature_dim),
                    torch.nn.LeakyReLU(0.1),
                    torch.nn.Linear(self.emb_feature_dim, self.emb_feature_dim),
                ),
                "pred": torch.nn.Sequential(
                    torch.nn.Linear(self.trans_feature_dim, self.emb_feature_dim),
                    # torch.nn.BatchNorm1d(self.emb_feature_dim),
                    torch.nn.LeakyReLU(0.1),
                    torch.nn.Linear(self.emb_feature_dim, self.emb_feature_dim),
                ),
                "obj": torch.nn.Sequential(
                    torch.nn.Linear(self.trans_feature_dim, self.emb_feature_dim),
                    # torch.nn.BatchNorm1d(self.emb_feature_dim),
                    torch.nn.LeakyReLU(0.1),
                    torch.nn.Linear(self.emb_feature_dim, self.emb_feature_dim),
                ),
            }
        )

        self.distance_function = lambda x, y: 1.0 - F.cosine_similarity(x, y)

        # Triplet Loss (Cosine Distance)
        self.triplet_loss = ModuleDict(
            {
                "subj": nn.TripletMarginWithDistanceLoss(
                    distance_function=self.distance_function,
                    margin=0.2,
                ),
                "pred": nn.TripletMarginWithDistanceLoss(
                    distance_function=self.distance_function,
                    margin=0.2,
                ),
                "obj": nn.TripletMarginWithDistanceLoss(
                    distance_function=self.distance_function,
                    margin=0.2,
                ),
            }
        )

    def _freeze_parameters(self, cfg):
        freeze_detectron = True
        if freeze_detectron:
            for param in self.detectron.parameters():
                param.requires_grad = False

            for param in self.backbone.parameters():
                param.requires_grad = False

    def _load_detectron_chkpoints(self, cfg):
        """
        Extension of __init__ for modules
        """
        # Load Detectron2 Pre-Trained Weights
        if cfg.VRD_RESNETS101_PRETRAINED_WEIGHTS is not None:
            DetectionCheckpointer(self.detectron).load(
                os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
            )

    def _load_words_chkpoints(self, cfg):
        """
        Extension of __init__ for modules
        """
        return get_triples_features(cfg.DATASETS.TRAIN[0].split("_")[0])

    def _save_words_chkpoints(self, cfg):
        torch.save(self.token_embeddings, self.triples_embeddings_path)

    def _get_bert_features(self, triples):
        """
        Args:
            triples: (Subj, Pred, Obj)
        Return:
            dict of [CLS, Subj, Pred, Obj, SEP] embeddings
        """
        results = {}

        # Load pre-trained model tokenizer (vocabulary)
        marked_text = "[CLS] " + " ".join(triples) + " [SEP]"
        tokenized_text = self.tokenizer.tokenize(marked_text)

        # Save the token split to average them later on
        token_placements = defaultdict(list)
        triples_temp = list(triples)
        for i, tok in enumerate(tokenized_text):
            stip_tok = tok.replace("#", "")
            if stip_tok in triples_temp[0]:
                token_placements["subj"].append(i)
                triples_temp[0] = triples_temp[0].replace(stip_tok, "")
            elif stip_tok in triples_temp[1]:
                token_placements["pred"].append(i)
                triples_temp[1] = triples_temp[1].replace(stip_tok, "")
            elif stip_tok in triples_temp[2]:
                token_placements["obj"].append(i)
                triples_temp[2] = triples_temp[2].replace(stip_tok, "")
            elif not tok == "[CLS]" and not tok == "[SEP]":
                print(tok, triples)

        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        segments_ids = [1] * len(tokenized_text)  # one sentence

        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor([indexed_tokens]).to("cuda")
        segments_tensors = torch.tensor([segments_ids]).to("cuda")

        # Put the model in "evaluation" mode, meaning feed-forward operation.
        with torch.no_grad():
            outputs = self.bert_model(tokens_tensor, segments_tensors)

            # Evaluating the model will return a different number of objects based on
            # how it's  configured in the `from_pretrained` call earlier. In this case,
            # becase we set `output_hidden_states = True`, the third item will be the
            # hidden states from all layers. See the documentation for more details:
            # https://huggingface.co/transformers/model_doc/bert.html#bertmodel
            hidden_states = outputs[2]

        token_embeddings = torch.stack(hidden_states, dim=0)
        token_embeddings.size()

        # remove dimension 1
        token_embeddings = torch.squeeze(token_embeddings, dim=1)
        token_embeddings = token_embeddings.permute(1, 0, 2)

        # get token embeddings (list of token embeddings)
        token_vecs_cat = []
        for token in token_embeddings:
            cat_vec = torch.cat((token[-1], token[-2], token[-3], token[-4]), dim=0)
            token_vecs_cat.append(cat_vec)
        results["CLS"] = token_vecs_cat[0]
        results["SEP"] = token_vecs_cat[-1]

        # average the token embeddings for word that are splitted to get word embeddings
        for key, val in token_placements.items():
            results[key] = token_vecs_cat[val[0]]
            for i in range(1, len(val)):
                results[key] += token_vecs_cat[val[i]]
            results[key] = results[key] / len(val)

        return results

    def _get_word_features(self, triples):
        """
        Args:
            triples: ([(subj, pred, obj)]) list of triple
        Return:
            resulting embeddings for subjs, preds, and objs
        """
        subj_embeddings = []
        pred_embeddings = []
        obj_embeddings = []

        for subj, pred, obj in triples:
            subj_cls_label = self.object_classes[subj]
            pred_cls_label = self.predicate_classes[pred]
            obj_cls_label = self.object_classes[obj]
            triples_text = "-".join((subj_cls_label, pred_cls_label, obj_cls_label))
            query_embeddings = None
            if triples_text not in self.token_embeddings:
                query_embeddings = self._get_bert_features(
                    (subj_cls_label, pred_cls_label, obj_cls_label)
                )
                self.token_embeddings[triples_text] = query_embeddings
            else:
                query_embeddings = self.token_embeddings[triples_text]
            subj_embeddings.append(query_embeddings["subj"].to("cuda"))
            pred_embeddings.append(query_embeddings["pred"].to("cuda"))
            obj_embeddings.append(query_embeddings["obj"].to("cuda"))

        subj_embeddings = torch.stack(subj_embeddings).to("cuda")
        pred_embeddings = torch.stack(pred_embeddings).to("cuda")
        obj_embeddings = torch.stack(obj_embeddings).to("cuda")

        return subj_embeddings, pred_embeddings, obj_embeddings

    def _get_word_predicate_features(self, subj, obj):
        """
        Input:
            subj: subject index
            obj: object index
        Output:
            A stack pred_embeddings for the above two subj and obj
        """
        subj_cls_label = self.object_classes[subj]
        obj_cls_label = self.object_classes[obj]

        # predicate label
        pred_embeddings = []
        subj_embeddings = []
        obj_embeddings = []
        for pred, pred_cls_label in enumerate(self.predicate_classes):
            triples_text = "-".join((subj_cls_label, pred_cls_label, obj_cls_label))
            query_embeddings = None
            if triples_text not in self.token_embeddings:
                query_embeddings = self._get_bert_features(
                    (subj_cls_label, pred_cls_label, obj_cls_label)
                )
                self.token_embeddings[triples_text] = query_embeddings
            else:
                query_embeddings = self.token_embeddings[triples_text]
            subj_embeddings.append(query_embeddings["subj"].to("cuda"))
            pred_embeddings.append(query_embeddings["pred"].to("cuda"))
            obj_embeddings.append(query_embeddings["obj"].to("cuda"))

        subj_embeddings = torch.stack(subj_embeddings).to("cuda")
        pred_embeddings = torch.stack(pred_embeddings).to("cuda")
        obj_embeddings = torch.stack(obj_embeddings).to("cuda")

        return subj_embeddings, pred_embeddings, obj_embeddings

    def get_instances_prediction(self, data):
        """
        This function gets the predicted instances from the object detector,
        and is only relevant to relationship detection evaluation.

        Input:
            data - data format for detectron2
        Output:
            instances - Instances object by the detectron2 that contain predicted instances
        """
        instances = []

        images = torch.unsqueeze(data["image"], axis=0).cuda().float()
        images_list = ImageList(images, [(800, 800)])
        self.detectron.eval()
        features = self.detectron.backbone(images)
        proposals, _ = self.detectron.proposal_generator(images_list, features)
        instances, _ = self.detectron.roi_heads(images, features, proposals)

        return instances

    def enumerate_relationships_from_instances(self, instances):
        """
        This function convert the format of instancesinto all possible combinations
        of relationships in the detectron2 format

        Input:
            instances - Instances object by detectron2
        Output:
            relationships - a dictionary of relationships in the vrdtransr input format
        """
        enumerated_relationships = {
            "subj_bboxes": [],
            "obj_bboxes": [],
            "union_bboxes": [],
            "subj_classes": [],
            "obj_classes": [],
            "subj_scores": [],
            "obj_scores": [],
        }

        # TO DO: implement to support batch images instead of a single image
        instance_dict = instances[0].get_fields()

        pred_bboxes = instance_dict["pred_boxes"]
        pred_cls = instance_dict["pred_classes"]
        pred_scores = instance_dict["scores"]

        for i in range(len(pred_cls)):
            for j in range(i + 1, len(pred_cls)):
                obj1_bbox = pred_bboxes[i].tensor.cpu().detach().numpy().astype(int)
                obj2_bbox = pred_bboxes[j].tensor.cpu().detach().numpy().astype(int)
                union_bbox = boxes_union(
                    copy.deepcopy(obj1_bbox), copy.deepcopy(obj2_bbox)
                )[0]

                obj1_label = pred_cls[i]
                obj2_label = pred_cls[j]

                obj1_score = pred_scores[i]
                obj2_score = pred_scores[j]

                enumerated_relationships["subj_bboxes"].append(obj1_bbox)
                enumerated_relationships["obj_bboxes"].append(obj2_bbox)
                enumerated_relationships["union_bboxes"].append(union_bbox)
                enumerated_relationships["subj_classes"].append(obj1_label)
                enumerated_relationships["obj_classes"].append(obj2_label)
                enumerated_relationships["subj_scores"].append(obj1_score)
                enumerated_relationships["obj_scores"].append(obj2_score)

                enumerated_relationships["subj_bboxes"].append(obj2_bbox)
                enumerated_relationships["obj_bboxes"].append(obj1_bbox)
                enumerated_relationships["union_bboxes"].append(union_bbox)
                enumerated_relationships["subj_classes"].append(obj2_label)
                enumerated_relationships["obj_classes"].append(obj1_label)
                enumerated_relationships["subj_scores"].append(obj2_score)
                enumerated_relationships["obj_scores"].append(obj1_score)

        return enumerated_relationships

    def get_predicted_relationships(self, data):
        """
        This function gets the predicted relationships from the object detector,
        and is only relevant to relationship detection evaluation.

        Input:
            data - data format for detectron2
        Output:
            relationships - Instances object by the detectron2 that contain predicted instances
        """
        # get predicted objects in the given image
        instances = self.get_instances_prediction(data)

        # enumerate the relationships with the predicted instances (bounding boxes and labels)
        relationships = self.enumerate_relationships_from_instances(instances)

        return relationships

    def get_predicate_distances(self, data, is_rel_eval=False):
        """
        Predict model's prediction based on the given data.
        Return the prediction predicate, visual relationship, and phrase (to be implemented)
        Input:
            data: vrdtranse input format
            is_rel_eval: boolean whether the evaluation is predicate detection or relationship detection
        """
        relationships = data["relationships"].copy()
        all_predicate_distances = (
            []
        )  # for each (subj, obj) pair, we get a set of distances
        all_subject_distances = []  # distance between subject visual and language
        all_object_distances = []
        all_predicate_subtract_distances = []
        all_transe_visual_feature = []
        all_transe_language_feature = []
        all_transe_distance_feature = []

        # forward features for gt_visual and gt_text
        fc_features = self.forward(data, None, get_fc_features=True)

        rel_cnt = 0
        with torch.no_grad():
            for subj, obj in zip(
                relationships["subj_classes"], relationships["obj_classes"]
            ):
                # compute all potential predicate embeddings for the (subj, obj) pair
                (
                    all_subj_embeddings,
                    all_predicate_embeddings,
                    all_object_embeddings,
                ) = self._get_word_predicate_features(subj=subj, obj=obj)

                # languge features
                fc_all_subject_embeddings = self.fc_word["subj"](all_subj_embeddings)
                fc_all_predicate_embeddings = self.fc_word["pred"](
                    all_predicate_embeddings
                )
                fc_all_object_embeddings = self.fc_word["obj"](all_object_embeddings)
                fc_pred_transe_language_feature = (
                    fc_all_subject_embeddings
                    + fc_all_predicate_embeddings
                    - fc_all_object_embeddings
                )

                # visual features
                fc_pred_visual_feature = fc_features["visual"]["pred"][rel_cnt, :]
                fc_pred_subtract_visual_feature = (
                    fc_features["visual"]["obj"][rel_cnt, :]
                    - fc_features["visual"]["subj"][rel_cnt, :]
                )
                fc_pred_transe_visual_feature = (
                    fc_features["visual"]["subj"][rel_cnt, :]
                    + fc_features["visual"]["pred"][rel_cnt, :]
                    - fc_features["visual"]["obj"][rel_cnt, :]
                )
                fc_subj_visual_feature = fc_features["visual"]["subj"][rel_cnt, :]
                fc_obj_visual_feature = fc_features["visual"]["obj"][rel_cnt, :]

                # compute distance between the fc_features["visual"]["pred"] and fc_predicate_embeddings to get top n
                pdist = lambda x, y: 1.0 - F.cosine_similarity(x, y)
                distance = []
                distance_subject = []
                distance_object = []
                distance_subtract = []
                distance_transe = []

                for subj_emb, pred_emb, obj_emb, pred_transe_emb in zip(
                    fc_all_subject_embeddings,
                    fc_all_predicate_embeddings,
                    fc_all_object_embeddings,
                    fc_pred_transe_language_feature,
                ):
                    distance.append(
                        pdist(
                            torch.unsqueeze(fc_pred_visual_feature, dim=0),
                            torch.unsqueeze(pred_emb, dim=0),
                        )
                    )
                    distance_subtract.append(
                        pdist(
                            torch.unsqueeze(fc_pred_subtract_visual_feature, dim=0),
                            torch.unsqueeze(obj_emb - subj_emb, dim=0),
                        )
                    )
                    distance_transe.append(
                        pdist(
                            torch.unsqueeze(fc_pred_transe_visual_feature, dim=0),
                            torch.unsqueeze(pred_transe_emb, dim=0),
                        )
                    )
                    distance_subject.append(
                        pdist(
                            torch.unsqueeze(fc_subj_visual_feature, dim=0),
                            torch.unsqueeze(subj_emb, dim=0),
                        )
                    )
                    distance_object.append(
                        pdist(
                            torch.unsqueeze(fc_obj_visual_feature, dim=0),
                            torch.unsqueeze(obj_emb, dim=0),
                        )
                    )

                # add set of distances to the given relationship
                all_predicate_distances.append(distance)
                all_subject_distances.append(distance_subject)
                all_object_distances.append(distance_object)
                all_predicate_subtract_distances.append(distance_subtract)
                all_transe_visual_feature.append(fc_pred_transe_visual_feature)
                all_transe_language_feature.append(fc_pred_transe_language_feature)
                all_transe_distance_feature.append(distance_transe)
                rel_cnt += 1

        if is_rel_eval:
            return (
                all_predicate_distances,
                all_subject_distances,
                all_object_distances,
                all_predicate_subtract_distances,
                all_transe_distance_feature,
            )

        return (
            all_predicate_distances,
            all_predicate_subtract_distances,
            all_transe_distance_feature,
        )

    def _get_word_object_features(self, subj, pred):
        """
        Input:
            subj: subject index
            pred: predicate index
        Output:
            A stack pred_embeddings for the above two subj and obj
        """
        subj_cls_label = self.object_classes[subj]
        pred_cls_label = self.predicate_classes[pred]

        # predicate label
        pred_embeddings = []
        subj_embeddings = []
        obj_embeddings = []
        for obj, obj_cls_label in enumerate(self.object_classes):
            triples_text = "-".join((subj_cls_label, pred_cls_label, obj_cls_label))
            query_embeddings = None
            if triples_text not in self.token_embeddings:
                query_embeddings = self._get_bert_features(
                    (subj_cls_label, pred_cls_label, obj_cls_label)
                )
                self.token_embeddings[triples_text] = query_embeddings
            else:
                query_embeddings = self.token_embeddings[triples_text]
            subj_embeddings.append(query_embeddings["subj"].to("cuda"))
            pred_embeddings.append(query_embeddings["pred"].to("cuda"))
            obj_embeddings.append(query_embeddings["obj"].to("cuda"))

        subj_embeddings = torch.stack(subj_embeddings).to("cuda")
        pred_embeddings = torch.stack(pred_embeddings).to("cuda")
        obj_embeddings = torch.stack(obj_embeddings).to("cuda")

        return subj_embeddings, pred_embeddings, obj_embeddings

    def get_object_distances(self, data):
        """
        Predict model's prediction based on the given data.
        Return the prediction predicate, phrase (to be implemented), and visual relationship (to be implemented)
        """
        relationships = data["relationships"]
        all_object_distances = (
            []
        )  # for each (subj, pred) pair, we get a set of distances
        all_object_add_distances = []

        # forward features for gt_visual and gt_text
        fc_features = self.forward(
            data, None, get_fc_features=True, obfuscate_object=True
        )

        rel_cnt = 0
        with torch.no_grad():
            for subj, pred in zip(
                relationships["subj_classes"], relationships["pred_classes"]
            ):
                # compute all potential predicate embeddings for the (subj, obj) pair
                (
                    all_subj_embeddings,
                    all_predicate_embeddings,
                    all_object_embeddings,
                ) = self._get_word_object_features(subj=subj, pred=pred)

                # languge features
                fc_all_subject_embeddings = self.fc_word["subj"](all_subj_embeddings)
                fc_all_predicate_embeddings = self.fc_word["pred"](
                    all_predicate_embeddings
                )
                fc_all_object_embeddings = self.fc_word["obj"](all_object_embeddings)

                # visual features
                fc_obj_add_visual_feature = (
                    fc_features["visual"]["subj"][rel_cnt, :]
                    + fc_features["visual"]["pred"][rel_cnt, :]
                )

                # compute distance between the fc_features["visual"]["pred"] and fc_predicate_embeddings to get top n
                pdist = lambda x, y: 1.0 - F.cosine_similarity(x, y)
                distance_add = []
                distance = []

                for subj_emb, pred_emb, obj_emb in zip(
                    fc_all_subject_embeddings,
                    fc_all_predicate_embeddings,
                    fc_all_object_embeddings,
                ):
                    distance.append(
                        pdist(
                            torch.unsqueeze(fc_obj_add_visual_feature, dim=0),
                            torch.unsqueeze(obj_emb, dim=0),
                        )
                    )
                    distance_add.append(
                        pdist(
                            torch.unsqueeze(fc_obj_add_visual_feature, dim=0),
                            torch.unsqueeze(subj_emb + pred_emb, dim=0),
                        )
                    )

                # add set of distances to the given relationship
                all_object_distances.append(distance)
                all_object_add_distances.append(distance_add)
                rel_cnt += 1

        return all_object_distances, all_object_add_distances

    def _get_prediced_bboxes(self, data):

        data["height"] = 800
        data["width"] = 800

        self.detectron.eval()
        with torch.no_grad():
            outputs = self.detectron([data])

        return outputs

    def _get_roi_features(self, images, box_lists):
        """
        Get image features from the backbone network
        Input:
            images: (ImageList.from_tensors) with dimension (C,W,H)
            box_lists: A list of N boxes
        Return:
            features:[N, 7*7*256]
        """
        N = len(box_lists[0])

        cfg = self.cfg
        feature_maps = self.backbone(images)
        feature_maps = [feature_maps["p{}".format(i)] for i in range(2, 6)]
        regions_feature = self.pooler(feature_maps, box_lists)
        return regions_feature.reshape((N, self.visual_feature_dim))

    def get_unrelated_instance(
        self, bbox, cls, gt_tuple_boxes, gt_classes, memo, other_memo=None
    ):
        negative_example = {}

        tuple_bbox = tuple(bbox)
        for i, neg_bbox in enumerate(gt_tuple_boxes):
            if (
                other_memo is not None
                and neg_bbox not in other_memo
                and neg_bbox not in memo
            ):
                negative_example = {
                    "bbox": torch.from_numpy(np.asarray(neg_bbox))
                    .float()
                    .cuda(),  # convert to tensor float
                    "cls": gt_classes[i],
                }
                return negative_example
            elif neg_bbox != tuple_bbox and neg_bbox not in memo:
                negative_example = {
                    "bbox": torch.from_numpy(np.asarray(neg_bbox))
                    .float()
                    .cuda(),  # convert to tensor float
                    "cls": gt_classes[i],
                }
                return negative_example

        return negative_example
 
    def generate_negative_examples(self, data, K=3):
        """
        for each triple relation in data, generate K negative examples

        return: [{
            'subj_bboxes': Boxes(tensor[[X,Y,X2,Y2],...])),
            'union_bboxes': Boxes(tensor(([[X,Y,X2,Y2],...])),
            'obj_bboxes': Boxes(tensor([[X,Y,X2,Y2],...])),
            'subj_classes': [cls_subj,...],
            'pred_classes': [cls_pred,...],
            'obj_classes': [cls_obj,...]
        }]
        """
        boxes = data["instances"].get_fields()["gt_boxes"]
        # ensure that there is relationship
        if len(boxes) == 0:
            return []

        gt_tuple_boxes = [
            tuple([ele.item() for ele in box]) for box in boxes
        ]  # convert ground truth boxes into tuples

        classes = data["instances"].get_fields()["gt_classes"]
        gt_classes = [int(item) for item in classes]

        # shuffle to random select first K
        zip_gt_data = list(zip(gt_tuple_boxes, gt_classes))
        random.shuffle(zip_gt_data)
        gt_tuple_boxes, gt_classes = zip(*zip_gt_data)

        relationships = data["relationships"]
        subj_boxes = relationships["subj_bboxes"]
        union_boxes = relationships["union_bboxes"]
        obj_boxes = relationships["obj_bboxes"]
        subj_classes = relationships["subj_classes"]
        pred_classes = relationships["pred_classes"]
        obj_classes = relationships["obj_classes"]

        # generate K negative examples
        neg_examples = []
        memo_subj = set()
        memo_obj = set()
        existed_predicates = dict(
            zip(
                [tuple(item) for item in data["relationships"]["union_bboxes"]],
                data["relationships"]["pred_classes"],
            )
        )

        for i in range(min(len(gt_tuple_boxes) - 1, K)):
            neg_ex = defaultdict(list)
            if (
                len(memo_subj) == len(gt_tuple_boxes) - 1
                or len(memo_obj) == len(gt_tuple_boxes) - 1
            ):
                break

            try:
                for j in range(
                    len(subj_boxes)
                ):  # iterate through the relationships in the image
                    # subj
                    subj_box = subj_boxes[j]
                    subj_cls = subj_classes[j]

                    # ISSUE: in the case where the number of object in the image is actually smaller than K, it is kind useless
                    unrelated_subj_instance = self.get_unrelated_instance(
                        subj_box[0],
                        subj_cls,
                        gt_tuple_boxes,
                        gt_classes,
                        memo=memo_subj,
                    )
                    neg_ex["subj_bboxes"].append(unrelated_subj_instance["bbox"])
                    neg_ex["subj_classes"].append(unrelated_subj_instance["cls"])

                    # obj
                    obj_box = obj_boxes[j]
                    obj_cls = obj_classes[j]
                    other_memo = set()
                    other_memo.add(tuple(subj_box[0]))
                    unrelated_obj_instance = self.get_unrelated_instance(
                        obj_box[0],
                        obj_cls,
                        gt_tuple_boxes,
                        gt_classes,
                        memo=memo_obj,
                        other_memo=other_memo,
                    )
                    neg_ex["obj_bboxes"].append(unrelated_obj_instance["bbox"])
                    neg_ex["obj_classes"].append(unrelated_obj_instance["cls"])

                    # pred
                    new_union_box = boxes_union(
                        copy.deepcopy(
                            unrelated_subj_instance["bbox"].reshape(1, 4).to("cpu")
                        ),
                        copy.deepcopy(
                            unrelated_obj_instance["bbox"].reshape(1, 4).to("cpu")
                        ),
                    )[0]
                    new_predicate_class = len(self.predicate_classes) - 1
                    if tuple(new_union_box) in existed_predicates:
                        new_predicate_class = existed_predicates[tuple(new_union_box)]
                    neg_ex["union_bboxes"].append(
                        torch.from_numpy(np.asarray(new_union_box)).float().cuda()
                    )
                    neg_ex["pred_classes"].append(new_predicate_class)

                for j in range(len(subj_boxes)):
                    memo_subj.add(tuple(neg_ex["subj_bboxes"][j].tolist()))
                    memo_obj.add(tuple(neg_ex["obj_bboxes"][j].tolist()))
            except:
                break

            # stack the bounding boxes
            neg_ex["subj_bboxes"] = Boxes(torch.stack(neg_ex["subj_bboxes"]))
            neg_ex["obj_bboxes"] = Boxes(torch.stack(neg_ex["obj_bboxes"]))
            neg_ex["union_bboxes"] = Boxes(torch.stack(neg_ex["union_bboxes"]))

            # append to memory
            neg_examples.append(neg_ex)

        return neg_examples

    def get_spatial_features(self, relationships, is_negative=False):
        """
        Args:
            data: see data definition in forward function
        Return:
            spatial_features: a tensor of spatial features containing coordinates of the bounding box
        """

        def spatial_delta(entity1, entity2):
            """
            entity1, entity2: [X,Y,X,Y]
            """

            width1, height1 = entity1[2] - entity1[0], entity1[3] - entity1[1]
            width2, height2 = entity2[2] - entity2[0], entity2[3] - entity2[1]

            delta_feat = [
                (entity1[0] - entity2[0]) / width2,
                (entity1[1] - entity2[1]) / height2,
                np.log(width1 / width2),
                np.log(height1 / height2),
            ]
            return delta_feat

        def spatial_coordinates(entity):
            """
            entity: [X,Y,X,Y]
            """
            width, height = entity[2] - entity[0], entity[3] - entity[1]
            coordinate_feat = [
                entity[0] / 800,
                entity[1] / 800,
                entity[2] / 800,
                entity[3] / 800,
                width * height / 800 * 800,
            ]
            return coordinate_feat

        spatial_features = []
        # iterate through every relationship pair and construct an array of spatial feature
        for subj_bbox, obj_bbox, union_bbox in zip(
            relationships["subj_bboxes"],
            relationships["obj_bboxes"],
            relationships["union_bboxes"],
        ):
            feat = []
            # XYXY
            if is_negative:
                feat.extend(spatial_delta(subj_bbox.cpu(), obj_bbox.cpu()))
                feat.extend(spatial_delta(subj_bbox.cpu(), union_bbox.cpu()))
                feat.extend(spatial_delta(union_bbox.cpu(), obj_bbox.cpu()))
                feat.extend(spatial_coordinates(subj_bbox.cpu()))
                feat.extend(spatial_coordinates(obj_bbox.cpu()))
            else:
                feat.extend(spatial_delta(subj_bbox[0], obj_bbox[0]))
                feat.extend(spatial_delta(subj_bbox[0], union_bbox))
                feat.extend(spatial_delta(union_bbox, obj_bbox[0]))
                feat.extend(spatial_coordinates(subj_bbox[0]))
                feat.extend(spatial_coordinates(obj_bbox[0]))

            spatial_features.append(torch.from_numpy(np.asarray(feat)).float().cuda())

        return torch.stack(spatial_features)

    def eval_phrase_detection(self, data):
        # add batch dim to image
        image = torch.unsqueeze(data["image"], axis=0).cuda()

        # get prediction bounding boxes
        output = self._get_prediced_bboxes(data)
        bboxes = output[0]["instances"].get_fields()["pred_boxes"]
        bboxes_features = self._get_roi_features(image.float(), box_lists=[bboxes])

        bboxes_classes = output[0]["instances"].get_fields()["pred_classes"]
        conf_score = output[0]["instances"].get_fields()["scores"]
        # iterate through every pairs and compute score
        return output, bboxes_features

    def forward(
        self,
        data,
        negative_examples,
        get_fc_features=False,
        obfuscate_object=False,
        **kwargs,
    ):
        """
        Args:
            data: {
                    #Detectron
                    'file_name': os.path.join(self.images_dir, img_name),
                    'image_id': int(img_name.split('.')[0]),
                    'annotations': list(unique_objects.values()),

                    #Relationships
                    'relationships': {
                        'subj_bboxes': subj_bboxes,
                        'obj_bboxes': obj_bboxes,
                        'union_bboxes': union_bboxes,

                        'subj_classes': subj_classes,
                        'pred_classes': pred_classes,
                        'obj_classes': obj_classes,
                    }
                }
            context: #any external data (not implemented as of current)
        """
        image = torch.unsqueeze(data["image"], axis=0).cuda()

        relationships = data["relationships"]

        subj_bboxes = Boxes([bbox[0] for bbox in relationships["subj_bboxes"]]).to(
            "cuda"
        )
        if obfuscate_object:
            union_bboxes = subj_bboxes
        else:
            union_bboxes = Boxes([bbox for bbox in relationships["union_bboxes"]]).to(
                "cuda"
            )
        obj_bboxes = Boxes([bbox[0] for bbox in relationships["obj_bboxes"]]).to("cuda")

        # ground_truth features
        gt_features = {"visual": {}, "textual": {}}

        # fully connected features
        fc_features = {"visual": {}, "textual": {}}

        # spatial features
        gt_spatial_features = self.fc_spatial(self.get_spatial_features(relationships))

        # extract visual features from backbone and ROIPool for n relations in the image
        gt_features["visual"]["subj"] = self._get_roi_features(
            image.float(), box_lists=[subj_bboxes]
        )
        gt_features["visual"]["pred"] = self._get_roi_features(
            image.float(), box_lists=[union_bboxes]
        )
        gt_features["visual"]["pred"] = torch.cat(
            (gt_features["visual"]["pred"], gt_spatial_features), 1
        )
        gt_features["visual"]["obj"] = self._get_roi_features(
            image.float(), box_lists=[obj_bboxes]
        )

        # fc visual (rois1 and rois2)
        fc_features["visual"]["subj"] = self.fc_rois["subj"](
            gt_features["visual"]["subj"]
        )
        fc_features["visual"]["pred"] = self.fc_rois["pred"](
            gt_features["visual"]["pred"]
        )
        fc_features["visual"]["obj"] = self.fc_rois["obj"](gt_features["visual"]["obj"])

        fc_features["visual"]["subj"] = self.fc_rois2["subj"](
            fc_features["visual"]["subj"]
        )
        fc_features["visual"]["pred"] = self.fc_rois2["pred"](
            fc_features["visual"]["pred"]
        )
        fc_features["visual"]["obj"] = self.fc_rois2["obj"](
            fc_features["visual"]["obj"]
        )

        if get_fc_features:
            return fc_features

        # extract word embeddings for n examples in the image
        word_embeddings = self._get_word_features(
            list(
                zip(
                    relationships["subj_classes"],
                    relationships["pred_classes"],
                    relationships["obj_classes"],
                )
            )
        )
        gt_features["textual"]["subj"] = word_embeddings[0]
        gt_features["textual"]["pred"] = word_embeddings[1]
        gt_features["textual"]["obj"] = word_embeddings[2]

        # fc word
        fc_features["textual"]["subj"] = self.fc_word["subj"](
            gt_features["textual"]["subj"]
        )
        fc_features["textual"]["pred"] = self.fc_word["pred"](
            gt_features["textual"]["pred"]
        )
        fc_features["textual"]["obj"] = self.fc_word["obj"](
            gt_features["textual"]["obj"]
        )

        print("fc_features_shape:", fc_features["textual"]["subj"])

        # Visual and Language Consistency losses triplet_loss(anchor, positive, negative)
        triplet_softmax_losses = {
            "subj": None,
            "pred": None,
            "obj": None,
        }  # , "transr": None}

        neg_features_list = []
        # NEGATIVE EXAMPLES
        for neg_relationships in negative_examples:

            neg_subj_boxes = neg_relationships["subj_bboxes"]
            neg_union_boxes = neg_relationships["union_bboxes"]
            neg_obj_boxes = neg_relationships["obj_bboxes"]

            # dictionary to store gt_features
            neg_features = {"visual": {}, "textual": {}}

            # spatial features
            neg_spatial_features = self.fc_spatial(
                self.get_spatial_features(neg_relationships, is_negative=True)
            )

            # extract visual features from backbone and ROIPool for n relations in the image
            neg_features["visual"]["subj"] = self._get_roi_features(
                image.float(), box_lists=[neg_subj_boxes]
            )
            neg_features["visual"]["pred"] = self._get_roi_features(
                image.float(), box_lists=[neg_union_boxes]
            )
            neg_features["visual"]["pred"] = torch.cat(
                (neg_features["visual"]["pred"], neg_spatial_features), 1
            )
            neg_features["visual"]["obj"] = self._get_roi_features(
                image.float(), box_lists=[neg_obj_boxes]
            )

            # extract word embeddings for n examples in the image
            neg_word_embeddings = self._get_word_features(
                list(
                    zip(
                        neg_relationships["subj_classes"],
                        neg_relationships["pred_classes"],
                        neg_relationships["obj_classes"],
                    )
                )
            )
            neg_features["textual"]["subj"] = neg_word_embeddings[0]
            neg_features["textual"]["pred"] = neg_word_embeddings[1]
            neg_features["textual"]["obj"] = neg_word_embeddings[2]

            # neg fc visual
            neg_fc_features = {"visual": {}, "textual": {}}
            neg_fc_features["visual"]["subj"] = self.fc_rois["subj"](
                neg_features["visual"]["subj"]
            )
            neg_fc_features["visual"]["pred"] = self.fc_rois["pred"](
                neg_features["visual"]["pred"]
            )
            neg_fc_features["visual"]["obj"] = self.fc_rois["obj"](
                neg_features["visual"]["obj"]
            )

            neg_fc_features["visual"]["subj"] = self.fc_rois2["subj"](
                neg_fc_features["visual"]["subj"]
            )
            neg_fc_features["visual"]["pred"] = self.fc_rois2["pred"](
                neg_fc_features["visual"]["pred"]
            )
            neg_fc_features["visual"]["obj"] = self.fc_rois2["obj"](
                neg_fc_features["visual"]["obj"]
            )

            # neg fc word
            neg_fc_features["textual"]["subj"] = self.fc_word["subj"](
                neg_features["textual"]["subj"]
            )
            neg_fc_features["textual"]["pred"] = self.fc_word["pred"](
                neg_features["textual"]["pred"]
            )
            neg_fc_features["textual"]["obj"] = self.fc_word["obj"](
                neg_features["textual"]["obj"]
            )

            neg_features_list.append(neg_fc_features)

        triplet_softmax_losses = {
            "subj": self.triplet_softmax_loss["subj"](fc_features, neg_fc_features),
            "pred": self.triplet_softmax_loss["pred"](fc_features, neg_fc_features),
            "obj": self.triplet_softmax_loss["obj"](fc_features, neg_fc_features),
        }
        return triplet_softmax_losses
