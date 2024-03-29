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

from utils import generate_random_color, transform_xcycwh_to_x1y1x2y2, filter_confidence, run_NMS, scale_coords, transform_x1y1x2y2_to_x1y1wh, imwrite, visualize_prediction, analyse_mAP_info

SEED = 42
torch.manual_seed(SEED)
TIMESTAMP = datetime.today().strftime("%Y-%m-%d_%H-%M")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#MEAN = 0.4333
#STD = 0.2194

#MEAN = np.array([0.485, 0.456, 0.406]) # RGB
#STD = np.array([0.229, 0.224, 0.225]) # RGB

def to_tensor(image):
    image = np.ascontiguousarray(image.transpose(2, 0, 1))
    return torch.from_numpy(image).float()


def to_image(tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    denorm_tensor = tensor.clone()
    for t, m, s in zip(denorm_tensor, mean, std):
        t.mul_(s).add_(m)
    denorm_tensor.clamp_(min=0, max=1.)
    denorm_tensor *= 255
    image = denorm_tensor.permute(1,2,0).numpy().astype(np.uint8)
    return image

@torch.no_grad()
def validate(class_list, color_list, mAP_filepath, dataloader, model, evaluator, epoch=0, save_result=False, conf_thres = 0.25, nms_thres = 0.6, img_log_dir="/workspace/storage/object-detection/yolov1/experiments/training-image"):
    
    model.eval()
    
    with open(mAP_filepath, mode='r') as f:
        json_file = json.load(f)
    
    cocoPred = []
    check_images, check_preds, check_results = [], [], []
    imageToid = json_file["imageToid"]

    for _, minibatch in enumerate(dataloader):
        filenames, images, shapes = minibatch[0], minibatch[1], minibatch[3]
        predictions = model(images.to(device))

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
        imwrite(str(img_log_dir +"/"+ f"EP-{epoch:03d}.jpg"), concat_result)

    if len(cocoPred) > 0:
        cocoPred = np.concatenate(cocoPred, axis=0)
        mAP_dict, eval_text = evaluator(predictions=cocoPred)

        if save_result:
            np.savetxt("/workspace/storage/object-detection/yolov1/experiments/predictions.txt", cocoPred, fmt="%.4f", delimiter=",", header=f"Inference results of [image_id, x1y1wh, score, label] on {TIMESTAMP}")
        return mAP_dict, eval_text
    else:
        return None, None

def result_analyis(class_list, mAP_dict, path="/workspace/storage/object-detection/yolov1/experiments"):
    analysis_result = analyse_mAP_info(mAP_dict, class_list)
    data_df, figure_AP, figure_dets, fig_PR_curves = analysis_result
    data_df.to_csv(str(path + f"/result-AP.csv"))
    figure_AP.savefig(str(path + f"/figure-AP.jpg"))
    figure_dets.savefig(str(path + f"/figure-dets.jpg"))
    PR_curve_dir = path + "/PR-curve" 
    os.makedirs(PR_curve_dir, exist_ok=True)
    for class_id in fig_PR_curves.keys():
        fig_PR_curves[class_id].savefig(str(PR_curve_dir + f"/{class_list[class_id]}.jpg"))
        fig_PR_curves[class_id].clf()
