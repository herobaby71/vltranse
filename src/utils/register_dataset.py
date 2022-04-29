from detectron2.data import DatasetCatalog, MetadataCatalog
from utils.annotations import get_object_classes, get_vrd_dicts


def register_vrd_dataset(set_name):
    """
    Register dataset and its metadata to the detectron2 engine
    Input:
        set_name: 'vrd' or vg200
    """
    thing_classes = get_object_classes(set_name)

    # register the annotations
    for d_type in ["train", "val"]:
        DatasetCatalog.register(
            "_".join((set_name, d_type)),
            lambda d_type=d_type: get_vrd_dicts("/".join((set_name, d_type))),
        )
        MetadataCatalog.get("_".join((set_name, d_type))).set(
            thing_classes=thing_classes
        )
