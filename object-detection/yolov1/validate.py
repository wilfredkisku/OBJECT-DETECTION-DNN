import os
import json
import pprint
import platform
import argparse
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from dataset import Dataset
from transform import BasicTransform, AugmentTransform
from yolo import YoloModel

from logger import build_basic_logger

from utils import generate_random_color, transform_xcycwh_to_x1y1x2y2, filter_confidence, run_NMS, scale_coords, transform_x1y1x2y2_to_x1y1wh, imwrite, visualize_prediction

SEED = 42
torch.manual_seed(SEED)
TIMESTAMP = datetime.today().strftime("%Y-%m-%d_%H-%M")

@torch.no_grad()
def validate(class_list, color_list, mAP_filepath, dataloader, model, evaluator, epoch=0, save_result=False, conf_thres = 0.001, nms_thres = 0.6):
    
    model.eval()
    
    with open(mAP_filepath, mode='r') as f:
        json_file = json.load(f)
    
    cocoPred = []
    check_images, check_preds, check_results = [], [], []
    imageToid = json_file["imageToid"]

    for _, minibatch in enumerate(dataloader):
        filenames, images, shapes = minibatch[0], minibatch[1], minibatch[3]
        predictions = model(images).to(device)

        for j in range(len(filenames)):
            prediction = predictions[j].cpu().numpy()
            prediction[:, 1:5] = transform_xcycwh_to_x1y1x2y2(boxes=prediction[:, 1:5], clip_max=1.0)
            prediction = filter_confidence(prediction=prediction, conf_threshold=conf_thres)
            prediction = run_NMS(prediction=prediction, iou_threshold=nms_thres)

            if len(check_images) < 5:
                check_images.append(to_image(images[j]))
                check_preds.append(prediction.copy())

            if len(prediction) > 0:
                filename = filenames[j]
                shape = shapes[j]
                cls_id = prediction[:, [0]]
                conf = prediction[:, [-1]]
                box_x1y1x2y2 = scale_coords(img1_shape=images.shape[2:], coords=prediction[:, 1:5], img0_shape=shape[:2])
                box_x1y1wh = transform_x1y1x2y2_to_x1y1wh(boxes=box_x1y1x2y2)
                img_id = np.array((imageToid[filename],) * len(cls_id))[:, np.newaxis]
                cocoPred.append(np.concatenate((img_id, box_x1y1wh, conf, cls_id), axis=1))

    del images, predictions

    if (epoch % 10 == 0) and img_log_dir:
        for k in range(len(check_images)):
            check_image = check_images[k]
            check_pred = check_preds[k]
            check_result = visualize_prediction(image=check_image, prediction=check_pred, class_list=class_list, color_list=color_list)
            check_results.append(check_result)
        concat_result = np.concatenate(check_results, axis=1)
        imwrite(str(args.img_log_dir / f"EP-{epoch:03d}.jpg"), concat_result)

    if len(cocoPred) > 0:
        cocoPred = np.concatenate(cocoPred, axis=0)
        mAP_dict, eval_text = evaluator(predictions=cocoPred)

        if save_result:
            np.savetxt(args.exp_path / "predictions.txt", cocoPred, fmt="%.4f", delimiter=",", header=f"Inference results of [image_id, x1y1wh, score, label] on {TIMESTAMP}")
        return mAP_dict, eval_text
    else:
        return None, None
