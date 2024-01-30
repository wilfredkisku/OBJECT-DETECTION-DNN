import os
import random
import torch

from logger import build_basic_logger
from dataset import Dataset
from transform import BasicTransform, AugmentTransform

from torch import nn
from torch import optim
from pathlib import Path
from torch.utils.data import DataLoader

#constants 
SEED = 42 

def main_work(logger):

    random.seed(SEED)
    torch.manual_seed(SEED)
    
    logging = logger

    global epoch

    train_dataset = Dataset(yaml_path=yaml_path, phase="train")
    train_transformer = AugmentTransform(input_size=input_size)
    train_dataset.load_transformer(transformer=train_transformer)
    train_loader = DataLoader(dataset=train_dataset, collate_fn=Dataset.collate_fn, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=workers)

    val_dataset = Dataset(yaml_path=yaml_path, phase="val")
    val_transformer = BasicTransform(input_size=input_size)
    val_dataset.load_transformer(transformer=val_transformer)
    val_loader = DataLoader(dataset=val_dataset, collate_fn=Dataset.collate_fn, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=workers)



if __name__ == "__main__":
    
    exp_path = "/home/wilfred/Desktop/object-detection/yolov3/experiments"
    
    logger = build_basic_logger(os.path.join(exp_path, "train.log"))
    main_work(logger=logger)
