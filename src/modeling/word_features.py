"""
    Get Word Features or <Subject, Predicate, Object> features
    Related Implementations:
        - https://pypi.org/project/pytorch-pretrained-bert/
"""
import torch
import os
import json

from collections import defaultdict
from config import TRIPLES_EMBEDDING_PATH
from utils.annotations import get_vrd_dicts, get_object_classes, get_predicate_classes
from transformers import BertTokenizer, BertModel


def get_word_features(triples, model, tokenizer):
    """
    Args:
        triples: (Subj, Pred, Obj)
    Return:
        dict of [CLS, Subj, Pred, Obj, SEP] embeddings
    """
    results = {}

    # Load pre-trained model tokenizer (vocabulary)
    marked_text = "[CLS] " + " ".join(triples) + " [SEP]"
    tokenized_text = tokenizer.tokenize(marked_text)

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

    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [1] * len(tokenized_text)  # one sentence

    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    # Put the model in "evaluation" mode, meaning feed-forward operation.
    model.eval()

    with torch.no_grad():
        outputs = model(tokens_tensor, segments_tensors)

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


def get_triples_features(set_name="vrd"):

    # object classes and predicate classes
    object_classes = get_object_classes(set_name)
    predicate_classes = get_predicate_classes(set_name)

    # check whether triples embeddings have been generated before
    triples_memo = {}
    if os.path.exists(TRIPLES_EMBEDDING_PATH):
        triples_memo = torch.load(TRIPLES_EMBEDDING_PATH)
        return triples_memo

    # initialize the model and tokenizer
    model = BertModel.from_pretrained(
        "bert-base-uncased",
        output_hidden_states=True,  # Whether the model returns all hidden-states.
    )
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # iterate through the all the triples, and extract the features
    dataset = get_vrd_dicts("vrd/train")
    for train_feat in dataset:
        rel = train_feat["relationships"]
        for subj_ind, pred_ind, obj_ind in zip(
            rel["subj_classes"], rel["pred_classes"], rel["obj_classes"]
        ):
            triples_text = (
                object_classes[subj_ind],
                predicate_classes[pred_ind],
                object_classes[obj_ind],
            )
            if "-".join(triples_text) in triples_memo:
                continue
            word_feat = get_word_features(triples_text, model, tokenizer)
            triples_memo["-".join(triples_text)] = word_feat

    try:
        torch.save(triples_memo, TRIPLES_EMBEDDING_PATH)
    except:
        pass

    return triples_memo


def get_trained_triples_memo(train_dataloader=None):
    """
    If there is no pre-computed memo, train_dataloader is required.
    Input:
        train_dataloader: dataloader
    """
    trained_triples_path = "../generated/trained_triples.json"
    trained_triples = {}
    if os.path.exists(trained_triples_path):
        with open(trained_triples_path, "r") as file:
            trained_triples = json.load(file)
    else:
        object_classes = get_object_classes()
        predicate_classes = get_predicate_classes()

        iter_dataloader = iter(train_dataloader)
        n_iters = len(train_dataloader.dataset.dataset)
        for i in range(n_iters):
            print(i)
            data = next(iter_dataloader)[0]
            relationships = data["relationships"]
            for j in range(len(relationships["subj_classes"])):
                subj_cls = object_classes[relationships["subj_classes"][j]]
                pred_cls = predicate_classes[relationships["pred_classes"][j]]
                obj_cls = object_classes[relationships["obj_classes"][j]]
                trained_triples["{}-{}-{}".format(subj_cls, pred_cls, obj_cls)] = 1

        with open(trained_triples_path, "w") as file:
            file.write(json.dumps(trained_triples))

    return trained_triples
