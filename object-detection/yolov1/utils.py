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

def visualize_prediction(image, prediction, class_list, color_list):
    input_size = image.shape[0]
    if len(prediction) > 0:
        prediction[:, 1:5] *= input_size
        image = visualize(image, prediction, class_list, color_list, show_class=True, show_score=True)
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

#########

matplotlib.use('Agg')
plt.rcParams.update({'figure.max_open_warning': 0})
TEXT_COLOR = (255, 255, 255)

def show_values(axs, orient='h', space=0.005, mode='ap'):
    def _single(ax):
        if orient == 'v':
            for p in ax.patches:
                _x = p.get_x() + p.get_width() / 2
                _y = p.get_y() + p.get_height() + (p.get_height()*0.01)
                if mode == 'ap':
                    value = f'{p.get_height():.2f}'
                else:
                    value = f'{p.get_height():.0f}'
                ax.text(_x, _y, value, ha='center') 
        elif orient == 'h':
            for p in ax.patches:
                _x = p.get_x() + p.get_width() + float(space)
                _y = p.get_y() + p.get_height() - (p.get_height()*0.3)
                if mode == 'ap':
                    value = f'{p.get_width():.2f}'
                else:
                    value = f'{p.get_width():.0f}'
                ax.text(_x, _y, value, ha='left')

    if isinstance(axs, np.ndarray):
        for _, ax in np.ndenumerate(axs):
            _single(ax)
    else:
        _single(axs)


def visualize_AP_per_class(data_df):
    plt.figure(figsize=(8, len(data_df)/4+4))
    ax = sns.barplot(x='AP_50', y='CATEGORY', data=data_df)
    ax.set(xlabel='AP@.50', ylabel='Category', title=f'AP@0.5 per categories (mAP@0.5: {data_df["AP_50"].mean():.4f})')
    show_values(ax, 'h', mode='ap')
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    plt.grid('on')
    fig = ax.get_figure()
    fig.tight_layout()
    return fig


def visualize_detect_rate_per_class(data_df):
    df_melt = pd.melt(data_df.drop('AP_50', axis=1), id_vars='CATEGORY', var_name='SOURCE', value_name='VALUE')
    plt.figure(figsize=(10, len(data_df)+2))
    ax = sns.barplot(x='VALUE', y='CATEGORY', hue='SOURCE', data=df_melt, palette='pastel')
    ax.set(xlabel='Count', ylabel='Category', title='Groundtruth & Detection')
    plt.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)
    show_values(ax, 'h', mode=None)
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    plt.grid('on')
    fig = ax.get_figure()
    fig.tight_layout()
    return fig


def visualize_PR_curve_per_class(pr_pts_per_class, class_list):
    fig_PR_curves = {}
    for class_id in pr_pts_per_class.keys():
        mprec = pr_pts_per_class[class_id]['mprec'][::-1][1:]
        mrec = pr_pts_per_class[class_id]['mrec'][::-1][1:]
        
        plt.figure(figsize=(4, 4))
        ax = sns.lineplot(x=mrec, y=mprec, estimator=None, sort=False)
        ax.set(xlabel='Recalls', ylabel='Precisions')
        ax.set_title(label=f'Category: {class_list[class_id]}', fontsize=12)
        plt.grid('on')
        fig = ax.get_figure()
        fig.tight_layout()
        fig_PR_curves[class_id] = fig
    return fig_PR_curves
    

def sort_dict(values_per_class):
    return dict(sorted(values_per_class.items()))


def analyse_mAP_info(mAP_info_area, class_list):
    AP_50_PER_CLASS = sort_dict(mAP_info_area['AP_50_PER_CLASS'])
    NUM_TP_50_PER_CLASS = sort_dict(mAP_info_area['NUM_TP_50_PER_CLASS'])
    NUM_FP_50_PER_CLASS = sort_dict(mAP_info_area['NUM_FP_50_PER_CLASS'])
    NUM_TRUE_PER_CLASS = sort_dict(mAP_info_area['NUM_TRUE_PER_CLASS'])
    PR_50_PTS_PER_CLASS = sort_dict(mAP_info_area['PR_50_PTS_PER_CLASS'])

    data_dict = {}
    for class_id in AP_50_PER_CLASS.keys():
        data_dict[class_id] = [class_list[class_id],
                               AP_50_PER_CLASS[class_id],
                               NUM_TRUE_PER_CLASS[class_id],
                               NUM_TP_50_PER_CLASS[class_id],
                               NUM_TRUE_PER_CLASS[class_id] - NUM_TP_50_PER_CLASS[class_id],
                               NUM_FP_50_PER_CLASS[class_id]]

    data_df = pd.DataFrame.from_dict(data=data_dict, orient='index', columns=['CATEGORY', 'AP_50', 'NUM_TRUE', 'NUM_TP', 'NUM_FN', 'NUM_FP'])
    figure_AP = visualize_AP_per_class(data_df)
    fig_PR_curves = visualize_PR_curve_per_class(PR_50_PTS_PER_CLASS, class_list)
    figure_detect_rate = visualize_detect_rate_per_class(data_df)
    return data_df, figure_AP, figure_detect_rate, fig_PR_curves


def visualize_class_dist(class_ids, class_list, rotation=60):
    class_ids = Counter(class_ids[class_ids >= 0])
    class_ids = sort_dict(class_ids)
    for k, v in class_ids.items():
        class_ids[k] = [class_list[k], v]
        
    data_df = pd.DataFrame.from_dict(data=class_ids, orient='index', columns=['CATEGORY', 'COUNT'])
    data_df = data_df.reset_index(drop=True)

    plt.figure(figsize=(len(data_df)/4+2, 6))
    ax = sns.barplot(x='CATEGORY', y='COUNT', data=data_df)
    ax.set(xlabel='Category', ylabel='Count', title='Category Distribution')
    plt.grid('on')
    plt.xticks(rotation=rotation)
    fig = ax.get_figure()
    fig.tight_layout()
    return fig
if __name__ == "__main__":
    #img_path = '/home/wilfred/Documents/DGX-BACKUP/data/PASCAL-VOC/archive/images'
    #srcpath = '/home/wilfred/Desktop/object-detection/yolov1/data/train_labels_persons'
    #dst = '/home/wilfred/Desktop/object-detection/yolov1/data/TrainImageFolder/images'
    #copyfiles(img_path, srcpath, dst)
    
    utils_visualize('/home/wilfred/Desktop/object-detection/yolov1/res/000000059635.jpg')
