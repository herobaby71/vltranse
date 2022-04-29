import sys, os
import torch
from torch.utils.data import DataLoader

from config import parse_args, get_vrd_cfg
from utils.register_dataset import register_vrd_dataset
from utils.trainer import CustomTrainer
from utils.dataset import VRDDataset
from modeling.reltransr import RelTransR

def finetune_detectron2():
    cfg = get_vrd_cfg()

    #Register Dataset (only vrd for now)
    register_vrd_dataset('vrd')

    #Finetune FasterRCNN Module
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = CustomTrainer(cfg) 
    trainer.resume_or_load(resume=True)
    trainer.train()
    
def main():
    args = parse_args()
    print('Called with args:')
    print(args)

    if not torch.cuda.is_available():
        sys.exit("Need a CUDA device to run the code.")

    train_dataset = VRDDataset(set_type='train')
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    cfg = get_vrd_cfg(args)
    model = RelTransR(cfg)
    #criterion = CUSTOM CRITERION GOES HERE MULTIPLE CRITERIONS WHICH ALTERNATIVELY COMPUTE
    #optimizer = OPTIMIZER CHOICE GOES HERE

if (__name__ == '__main__'):
    main()
