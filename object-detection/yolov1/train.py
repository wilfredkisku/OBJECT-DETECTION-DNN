import os
import sys
import random
import argparse
from pathlib import Path
from copy import deepcopy
from datetime import datetime
from collections import defaultdict

import torch
from thop import profile
import numpy as np
from tqdm import tqdm, trange
from torch import optim
from torch.utils.data import DataLoader, distributed

from dataset import Dataset
from transform import BasicTransform, AugmentTransform
from yolo import YoloModel
from loss import YoloLoss
from logger import build_basic_logger
from utils import generate_random_color
from evaluate import Evaluator

#constants
image_size = input_size = 128
batch_size = 64
num_epochs = 150
momentum = 0.9
base_lr = 0.001
weight_decay = 5e-4
lr_decay =  [90, 120]
label_smoothing = 0.1
conf_thres = 0.001
nms_thres = 0.6
workers = 1
SEED = 42

def train(args, dataloader, model, criterion, optimizer):
    loss_type = ["multipart", "obj", "noobj", "box", "cls"]
    losses = defaultdict(float)
    
    #to train
    model.train()
    #make gradients zero
    optimizer.zero_grad()

    #load minibatch 
    for i, minibatch in enumerate(dataloader):
        ni = i + len(dataloader) * (epoch - 1)
        if ni <= args.nw:
            args.grad_accumulate = max(1, np.interp(ni, [0, args.nw], [1, args.nominal_batch_size / args.batch_size]).round())
            set_lr(optimizer, args.base_lr * pow(ni / (args.nw), 4))

        images, labels = minibatch[1], minibatch[2]

        with amp.autocast(enabled=not args.no_amp):
            predictions = model(images.cuda(args.rank, non_blocking=True))
            loss = criterion(predictions=predictions, labels=labels)
        scaler.scale((loss[0] / args.grad_accumulate) * args.world_size).backward()

        if ni - args.last_opt_step >= args.grad_accumulate:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            if ema is not None:
                ema.update(model)
            args.last_opt_step = ni

        for loss_name, loss_value in zip(loss_type, loss):
            if not torch.isfinite(loss_value) and loss_name != "multipart":
                print(f"############## {loss_name} Loss is Nan/Inf ! {loss_value} ##############")
                sys.exit(0)
            else:
                losses[loss_name] += loss_value.item()

    del images, predictions
    torch.cuda.empty_cache()

    loss_str = f"[Train-Epoch:{epoch:03d}] "
    for loss_name in loss_type:
        losses[loss_name] /= len(dataloader)
        loss_str += f"{loss_name}: {losses[loss_name]:.4f}  "
    return loss_str


def main_task(yaml_path, logger):
    
    device = 'cpu'
    random.seed(SEED)
    torch.manual_seed(SEED)

    global epoch
    
    train_dataset = Dataset(yaml_path=yaml_path, phase="train")
    train_transformer = AugmentTransform(input_size=input_size)
    train_dataset.load_transformer(transformer=train_transformer)
    train_loader = DataLoader(dataset=train_dataset, collate_fn=Dataset.collate_fn, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=workers)

    val_dataset = Dataset(yaml_path=yaml_path, phase="val")
    val_transformer = BasicTransform(input_size=input_size)
    val_dataset.load_transformer(transformer=val_transformer)
    val_loader = DataLoader(dataset=val_dataset, collate_fn=Dataset.collate_fn, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=workers)
    
    class_list = train_dataset.class_list
    color_list = generate_random_color(len(class_list))
    mAP_filepath = val_dataset.mAP_filepath

    print(class_list, color_list, mAP_filepath)

    model = YoloModel(input_size=input_size, num_classes=1, pretrained=True).to(device)
    #macs, params = profile(deepcopy(model))
    criterion = YoloLoss(grid_size=model.grid_size, label_smoothing=label_smoothing)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_decay, gamma=0.1)
    evaluator = Evaluator(annotation_file=mAP_filepath)
    #print(macs, params)
    '''
    start_epoch = 1
    progress_bar = trange(start_epoch, num_epochs+1, total=num_epochs, intital=start_epoch, ncols=115)
    best_epoch, best_score, best_mAP_str, mAP_dict = 0, 0, "", None

    for epoch in progress_bar:
        
        train_loader = tqdm(train_loader, desc=f"[TRAIN:{epoch:03d}/{args.num_epochs:03d}]", ncols=115, leave=False)
        train_loss_str = train(args=args, dataloader=train_loader, model=model, criterion=criterion, optimizer=optimizer)
    '''

if __name__ == "__main__":

    yaml_path = "voc_person.yaml"
    
    logger_path = os.path.join(os.getcwd(), "experiments", "train_7_11_23_.log")
    logger = build_basic_logger(logger_path)
    
    #rank=0, world_size=1, args=args, logger=logger
    main_task(yaml_path, logger=logger)
