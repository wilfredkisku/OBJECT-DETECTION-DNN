import os
import cv2
import torch
import numpy as np

import random
from collections import Counter

import matplotlib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#tranform center [to] xy and vice-versa
def transform_xcycwh_to_x1y1x2y2(boxes, clip_max=None):
    x1y1 = boxes[:, :2] - boxes[:, 2:] / 2
    x2y2 = boxes[:, :2] + boxes[:, 2:] / 2
    x1y1x2y2 = np.concatenate((x1y1, x2y2), axis=1)
    return x1y1x2y2.clip(min=0, max=clip_max if clip_max is not None else 1)

def transform_x1y1x2y2_to_xcycwh(boxes):
    wh = boxes[:, 2:] - boxes[:, :2]
    xcyc = boxes[:, :2] + wh / 2
    return np.concatenate((xcyc, wh), axis=1)

#visualize
def generate_random_color(num_colors):
    color_list = []
    for i in range(num_colors):
        hex_color = ''.join([random.choice('0123456789ABCDEF') for j in range(6)])
        rgb_color = tuple(int(hex_color[k:k+2], 16) for k in (0, 2, 4))
        color_list.append(rgb_color)
    return color_list

def visualize_box(image, label, class_list, color_list, show_class=False, show_score=False, fontscale=0.7, thickness=2):
    class_id = int(label[0])
    box = label[1:5].astype(int)
    if label[0] >= 0:
        color = color_list[class_id]
        x_min, y_min, x_max, y_max = box
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)

        if show_class:
            class_name = class_list[class_id]
            if show_score:
                class_name += f'({label[-1]*100:.0f}%)'
            ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, fontscale, 2)
            cv2.rectangle(image, (x_min, y_min - int(fontscale*2 * text_height)), (x_min + text_width, y_min), color, -1)
            cv2.putText(image, text=class_name, org=(x_min, y_min - int((1-fontscale) * text_height)),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=fontscale, color=TEXT_COLOR, lineType=cv2.LINE_AA)
    return image

def visualize(image, label, class_list, color_list, show_class=False, show_score=False):
    canvas = image.copy()
    for item in label:
        canvas = visualize_box(canvas, item, class_list, color_list, show_class=show_class, show_score=show_score)
    return canvas[..., ::-1]

def visualize_target(image, label, class_list, color_list):
    img_h, img_w, _ = image.shape
    label_xcycwh = label.copy()
    label_xcycwh[:, 1:5] = transform_xcycwh_to_x1y1x2y2(label_xcycwh[:, 1:5])
    label_xcycwh[:, 1:5] = scale_to_original(label_xcycwh[:, 1:5], scale_w=img_w, scale_h=img_h)
    image = visualize(image, label_xcycwh, class_list, color_list, show_class=True, show_score=False)
    return image

#grid set
def scale_coords(img1_shape, coords, img0_shape):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
    pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    coords = scale_to_original(boxes=coords, scale_w=img1_shape[1], scale_h=img1_shape[0])
    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords

def clip_coords(boxes, shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2

def set_grid(grid_size):
    grid_y, grid_x = torch.meshgrid((torch.arange(grid_size), torch.arange(grid_size)), indexing="ij")
    return (grid_x, grid_y)

def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def scale_to_original(boxes, scale_w, scale_h):
    boxes[:,[0,2]] *= scale_w
    boxes[:,[1,3]] *= scale_h
    return boxes.round(2)

def scale_to_norm(boxes, image_w, image_h):
    boxes[:,[0,2]] /= image_w
    boxes[:,[1,3]] /= image_h
    return boxes

def transform_x1y1x2y2_to_x1y1wh(boxes):
    x1y1 = boxes[:, :2]
    wh = boxes[:, 2:] - boxes[:, :2]
    return np.concatenate((x1y1, wh), axis=1)

def transform_xcycwh_to_x1y1wh(boxes):
    x1y1 = boxes[:, :2] - boxes[:, 2:] / 2
    wh = boxes[:, 2:]
    return np.concatenate((x1y1, wh), axis=1).clip(min=0)

def filter_confidence(prediction, conf_threshold=0.01):
    keep = (prediction[:, 0] > conf_threshold)
    conf = prediction[:, 0][keep]
    box = prediction[:, 1:5][keep]
    cls_id = prediction[:, 5][keep]
    return np.concatenate([cls_id[:, np.newaxis], box, conf[:, np.newaxis]], axis=-1)

def hard_NMS(boxes, scores, iou_threshold):
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(1e-10, xx2 - xx1)
        h = np.maximum(1e-10, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    return keep

def run_NMS(prediction, iou_threshold, maxDets=100):
    keep = np.zeros(len(prediction), dtype=np.int)
    for cls_id in np.unique(prediction[:, 0]):
        inds = np.where(prediction[:, 0] == cls_id)[0]
        if len(inds) == 0:
            continue
        cls_boxes = prediction[inds, 1:5]
        cls_scores = prediction[inds, 5]
        cls_keep = hard_NMS(boxes=cls_boxes, scores=cls_scores, iou_threshold=iou_threshold)
        keep[inds[cls_keep]] = 1
    prediction = prediction[np.where(keep > 0)]
    order = prediction[:, 5].argsort()[::-1]
    return prediction[order[:maxDets]]

def imwrite(filename, img):
    try:
        ext = os.path.splitext(filename)[1]
        result, n = cv2.imencode(ext, img)

        if result:
            with open(filename, mode='w+b') as f:
                n.tofile(f)
            return True
        else:
            return False
    except Exception as e:
        print(e)
        return False

def copyfiles(imgpath, srcpath, dst):
    
    srclst = os.listdir(srcpath)

    import shutil
    
    for src in srclst:
        print('File copied :: '+src)
        shutil.copy(imgpath+'/'+src[:-4]+'.jpg', dst)
    
    return None
############
def utils_visualize(img_path):
    #label = np.array([[0.48404687499999993, 0.52875, 0.1389375, 0.6580833333333334]])
    #coord = transform_xcycwh_to_x1y1x2y2(label)
    
    x1,y1,w,h = int(416.93), int(37.46), 19.63, 60.13
    
    x2 = x1 + int(w)
    y2 = y1 + int(h)

    img = cv2.imread(img_path)
    
    color = (255, 0, 0)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
    
    import matplotlib.pyplot as plt

    plt.imshow(img)
    plt.show()

    #cv2.imshow("test", img) 
    #cv2.waitKey(0) 
    #cv2.destroyAllWindows() 
    return None
############
if __name__ == "__main__":
    #img_path = '/home/wilfred/Documents/DGX-BACKUP/data/PASCAL-VOC/archive/images'
    #srcpath = '/home/wilfred/Desktop/object-detection/yolov1/data/train_labels_persons'
    #dst = '/home/wilfred/Desktop/object-detection/yolov1/data/TrainImageFolder/images'
    #copyfiles(img_path, srcpath, dst)
    
    utils_visualize('/home/wilfred/Desktop/object-detection/yolov1/res/000000059635.jpg')
