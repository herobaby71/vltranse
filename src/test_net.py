import os

from config import parse_args, get_vrd_cfg
from utils.register_dataset import register_vrd_dataset
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.checkpoint import DetectionCheckpointer


def main():

    #Parse Arguments from Input
    #args = parse_args()
    cfg = get_vrd_cfg()

    #Register Dataset (only vrd for now)
    register_vrd_dataset('vrd')

    #Finetune FasterRCNN Module
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    output_dir = os.path.join(cfg.OUTPUT_DIR,"inference")
    predictor = DefaultPredictor(cfg) 
    evaluator = COCOEvaluator("vrd_val")
    val_loader = build_detection_test_loader(cfg, "vrd_val")
    print(inference_on_dataset(trainer.model, val_loader, evaluator))

if (__name__ == '__main__'):
    main()
