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
#from transform import BasicTransform, AugmentTransform
from Transform import BasicTransform, AugmentTransform
from yolo import YoloModel
from loss import YoloLoss

#checked
from logger import build_basic_logger
from matplotlib import pyplot as plt
from utils import generate_random_color
from evaluate import Evaluator
from validate import result_analyis, validate

#TRAINING PARAMETERS
DEVICE          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMAGE_SIZE      = INPUT_SIZE = 128
BATCH_SIZE      = 16
NUM_EPOCHS      = 300
MOMENTUM        = 0.9
BASE_LR         = 0.001
WEIGHT_DECAY    = 5e-4
LR_DECAY        = [90, 120]
LABEL_SMOOTHING = 0.1
CONF_THRES      = 0.25 #0.001
NMS_THRES       = 0.6
WORKERS         = 4 
SEED            = 42

WEIGHT_DIR = os.path.join("/home/wilfred/Desktop/dgx/object-detection/yolov1", "training")

def train(dataloader, model, criterion, optimizer):
    #loss types 
    loss_type = ["multipart", "obj", "noobj", "box", "cls"]
    losses = defaultdict(float)
    
    #to train
    model.train()

    #make gradients zero
    optimizer.zero_grad()

    #load minibatch 
    for i, minibatch in enumerate(dataloader):
        
        #set_lr(optimizer, args.base_lr * pow(ni / (args.nw), 4))
        # filenames, images, labels, size
        images, labels = minibatch[1], minibatch[2]
        
        #predictions + loss calculation
        predictions = model(images.to(device))
        loss = criterion(predictions=predictions, labels=labels)
        
        loss[0].backward()
        optimizer.step()
        optimizer.zero_grad()

        for loss_name, loss_value in zip(loss_type, loss):
            if not torch.isfinite(loss_value) and loss_name != "multipart":
                print(f'############## {loss_name} Loss is Nan/Inf ! {loss_value} ##############')
                sys.exit(0)
            else:
                losses[loss_name] += loss_value.item()

    del images, predictions
    #torch.cuda.empty_cache()

    loss_str = f"[Train-Epoch:{epoch:03d}] "
    
    for loss_name in loss_type:
        losses[loss_name] /= len(dataloader)
        loss_str += f"{loss_name}: {losses[loss_name]:.4f} "
    
    return loss_str

def main_task(yaml_path, logger):
    
    #DEVICE = "cpu/cuda"
    random.seed(SEED)
    torch.manual_seed(SEED)
    
    logging = logger

    global epoch
    
    train_dataset = Dataset(yaml_path=yaml_path, phase="train")
    train_transformer = AugmentTransform(input_size=INPUT_SIZE)
    train_dataset.load_transformer(transformer=train_transformer)
    train_loader = DataLoader(dataset=train_dataset, collate_fn=Dataset.collate_fn, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=WORKERS)

    val_dataset = Dataset(yaml_path=yaml_path, phase="val")
    val_transformer = BasicTransform(input_size=INPUT_SIZE)
    val_dataset.load_transformer(transformer=val_transformer)
    val_loader = DataLoader(dataset=val_dataset, collate_fn=Dataset.collate_fn, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=WORKERS)
    
    '''
    #examples = next(iter(train_loader))
    #print(examples)
    #plt.imshow(examples[1][0].numpy().reshape(128, 128, -1))
    #from Transform import to_image, denormalize
    #import cv2
    #print(examples[1][10].shape)
    #img = to_image(examples[1][10])
    #img_denorm = denormalize(examples[1][10])
    #print(img.shape)
    #cv2.imwrite("/home/wilfred/Desktop/test-norm.jpg", img)
    #cv2.imwrite("/home/wilfred/Desktop/test-denorm.jpg", img_denorm)
    #print(img.shape)
    #plt.imshow(img)
    #plt.show()
    
    #for tuple_ in examples:
    #    #plt.imshow(img.permute(1,2,0))
    #    #plt.show()
    #    #print(f"Label: {label}")
    #    print(tuple_[0])
    #    break
    '''

    #DENORMALIZE
    #VALIDATE
    class_list = train_dataset.class_list
    color_list = generate_random_color(len(class_list))
    mAP_filepath = val_dataset.mAP_filepath
        
    model = YoloModel(input_size=INPUT_SIZE, num_classes=1, pretrained=True).to(DEVICE)
    #macs, params = profile(deepcopy(model), inputs=(torch.randn(1, 3, image_size, image_size).to(device)), verbose=False)
    #print(int(macs), int(params))
    criterion = YoloLoss(grid_size=model.grid_size, label_smoothing=LABEL_SMOOTHING)
    #print(model.grid_size)
    optimizer = optim.SGD(model.parameters(), lr=BASE_LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=LR_DECAY, gamma=0.1)
    
    #EVALUATE
    evaluator = Evaluator(annotation_file=mAP_filepath)
    start_epoch = 1

    '''
    #progress_bar = trange(start_epoch, num_epochs+1, total=num_epochs, intital=start_epoch, ncols=115)
    progress_bar = range(start_epoch, num_epochs+1)
    best_epoch, best_score, best_mAP_str, mAP_dict = 0, 0, "", None
    
    #if True:
    #    return 0

    for epoch in progress_bar:   
        train_loader = tqdm(train_loader, desc=f"[TRAIN:{epoch:03d}/{num_epochs:03d}]", ncols=115, leave=False)
        train_loss_str = train(dataloader=train_loader, model=model, criterion=criterion, optimizer=optimizer)

        logging.warning(train_loss_str)
        save_opt = {"running_epoch": epoch,
                    "class_list": class_list,
                    "model_state": deepcopy(model).state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict()}
        torch.save(save_opt, weight_dir + "/last.pt")
    
        if epoch % 10 == 0:
            val_loader = tqdm(val_loader, desc=f"[VAL:{epoch:03d}/{num_epochs:03d}]", ncols=115, leave=False)
            mAP_dict, eval_text = validate(class_list=class_list, color_list=color_list, mAP_filepath=mAP_filepath, dataloader=val_loader, model=model, evaluator=evaluator, epoch=epoch, conf_thres = conf_thres)
            ap50 = mAP_dict["all"]["mAP_50"]
            logging.warning(eval_text)

            if ap50 > best_score:
                #check the args required
                result_analyis(class_list, mAP_dict=mAP_dict["all"])
                best_epoch, best_score, best_mAP_str = epoch, ap50, eval_text
                torch.save(save_opt, weight_dir + "/best.pt")

        scheduler.step()

    logging.warning(f"[Best mAP at {best_epoch}]{best_mAP_str}")
    '''

if __name__ == "__main__":

    yaml_path = "/home/wilfred/Desktop/dgx/object-detection/yolov1/voc_person.yaml"
    
    #LOGGER PATH
    logger_path = os.path.join('/home/wilfred/Desktop/dgx/object-detection/yolov1', "experiments", "train_24_1_24.log")
    logger = build_basic_logger(logger_path)

    
    # RANK          = 0         --> "Process id for computation"
    # WORLD_SIZE    = 1         --> "Number of available GPU devices"
    # LOGGER        = LOGGER    --> object returned for handling logging
    main_task(yaml_path, logger=logger)
