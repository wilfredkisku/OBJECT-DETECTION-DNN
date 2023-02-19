import os
import cv2
import copy
import glob
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from main import *


def displayImages(pred_items, gt_items):
    
    files = pred_items.keys()
    files = list(files)
    #print(files)
    

    for i in range(len(files)):
        img = cv2.imread(files[i][:-4]+'.jpg', 0)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        for pred_loc in pred_items[files[i][:-4]+'.jpg']:
            cv2.rectangle(img, (pred_loc[1],pred_loc[2]), (pred_loc[3],pred_loc[4]), (0,255,0), 1)
    
        for gt_loc in gt_items[files[i][:-4]+'.xml']:
            cv2.rectangle(img, (gt_loc[0],gt_loc[1]), (gt_loc[2],gt_loc[3]), (0,0,255), 1)

        img = cv2.putText(img, files[i][:-4]+'.jpg', (15,15), cv2.FONT_HERSHEY_SIMPLEX, .50, (255,0,255), 1, cv2.LINE_AA)
        cv2.imshow("Predictions", img)
        key = cv2.waitKey(0)

    return None

def sortLists(bbox_pred, bbox_gt):

    #testing and sorting the bbox_gt
    myKeys_gt = list(bbox_gt.keys())
    myKeys_gt.sort()
    sorted_bbox_gt = {i: bbox_gt[i] for i in myKeys_gt}

    myKeys_pred = list(bbox_pred.keys())
    myKeys_pred.sort()
    sorted_bbox_pred = {i: bbox_pred[i] for i in myKeys_pred}

    return sorted_bbox_pred, sorted_bbox_gt

def avgPWR(pred, gt):

    height = 288
    width = 352

    gt_pwr = {}
    pred_pwr = {}

    for gt_keys in gt.keys():
        
        xmax = 352
        ymax = 288
        xmin = 0
        ymin = 0

        for gt_i in gt[gt_keys]:
            if gt_i[0] <= xmax:
                xmax = gt_i[0]
            if gt_i[1] <= ymax:
                ymax = gt_i[1]
            if gt_i[2] >= xmin:
                xmin = gt_i[2]
            if gt_i[3] >= ymin:
                ymin = gt_i[3]
        
        gt_pwr[gt_keys] = ((xmin-xmax)*(ymin-ymax))/(height*width)
        #print(((xmin-xmax)*(ymin-ymax))/(height*width))
            
    for pred_keys in pred.keys():

        xmax = 352
        ymax = 288
        xmin = 0
        ymin = 0

        for pred_i in pred[pred_keys]:
            if pred_i[1] <= xmax:
                xmax = pred_i[1]
            if pred_i[2] <= ymax:
                ymax = pred_i[2]
            if pred_i[3] >= xmin:
                xmin = pred_i[3]
            if pred_i[4] >= ymin:
                ymin = pred_i[4]
        
        pred_pwr[pred_keys] = ((xmin-xmax)*(ymin-ymax))/(height*width)
        #print(((xmin-xmax)*(ymin-ymax))/(height*width))
    
    for g,p in zip(gt_pwr, pred_pwr):
        print(gt_pwr[g],pred_pwr[p])

    return None

'''
    calculateTPFP:
        process gt and pred dicts for obtaining the dataframe that contains information for evaluation of 
        TP and FP rates.

    Returns:
        dataframe df | format for df is {'filename':[...], 'confidence':[...], 'tp':[...], 'fp':[...]}
'''
def calculateTPFPdict(pred, gt):
    data = {'filename':[], 'confidence':[], 'tp':[], 'fp':[]}

    print(pred)
    print(gt)
    #df = pd.DataFrame.from_dict(data)
    
    return None


'''
    processList:
        process the bounding boxes for prediction into a dict

    Returns:
        bbox_pred | dict of predictions {'filename':[confidence-score, xmin, ymin, xmax, ymax], ...}
'''
def processList(bbox):
    
    bbox_pred_list = []
    bbox_pred = {}

    height = 288
    width = 352

    for bb in bbox:
        bbox_pred_list.append([bb[1][12:],bb[2:]])

    for bb in bbox_pred_list:
        if len(bb[1:][0]) != 0: 
            tmp = []
            
            for i in range(len(bb[1:][0])):
                
                score = float(bb[1:][0][i].split()[5])
                ymin = int(float(bb[1:][0][i].split()[-4][1:])*height)
                xmin = int(float(bb[1:][0][i].split()[-3])*width)
                ymax = int(float(bb[1:][0][i].split()[-2])*height)
                xmax = int(float(bb[1:][0][i].split()[-1][:-1])*width)

                #xmin = int(float(bb[1:][0][i].split()[-4][1:])*height)
                #ymin = int(float(bb[1:][0][i].split()[-3])*width)
                #xmax = int(float(bb[1:][0][i].split()[-2])*height)
                #ymax = int(float(bb[1:][0][i].split()[-1][:-1])*width)

                tmp.append([score, xmin, ymin, xmax, ymax])

            bbox_pred[bb[0]] = tmp
        else:
            bbox_pred[bb[0]] = bb[1:][0]

    return bbox_pred

"""
    parseListsPred:
        The method reformats the lists iteratively to extract the bbox infomration on theimages,
        it accepts a list of list that contains the bbox_file infomration in a string format.
        [['no-of-person-dets', 'file-name',[det1],[det2],...],...]
        
    Returns:
        (dict) bbox_pred | the dictinary as the filenames as lists and bbox_pred as lists for the predicted 
        information for a particular file.
            {'key':[[bbox_pred1],[bbox_pred2],...],...}
"""
def parseListsPred(bbox_file):
    
    with open(bbox_file) as fp:
        bbox_lines = fp.readlines()

    new_bbox_lines = [line.strip() for line in bbox_lines]
    tmp = []
    bbox_pred = []
    for b in new_bbox_lines:
        if b != '':
            tmp.append(b)
        else:
            bbox_pred.append(tmp)
            tmp = []

    bbox_pred.append(tmp)
    bbox_pred = processList(bbox_pred)

    return bbox_pred

"""
    parseXMLfiles:
        The method evaluates the xmlFile list iteratively to generate the bbox of ground truths
        it extracts the bbox information [xmin, ymin, xmax, ymax]. The xml files only contains the
        'person' class, rest are taken to be as background.

    Returns:
        (dict) bbox_gt | a dictionary with the filenames as keys and the bbox information as a list
            {'key':[[bbox_gt1],[bbox_get2],...],...}
"""
def parseXMLfiles(xmlfile):
    
    bbox_gt = {}

    for xf in xmlfile:
        xfile = xf[:-3]+'xml'
        tree = ET.parse(xfile)
        root = tree.getroot()

        tags = [elem.tag for elem in root.iter()]

        xmin, xmax, ymin, ymax = [],[],[],[]
        for xmi in root.iter('xmin'):
            xmin.append(xmi.text)
        for xma in root.iter('xmax'):
            xmax.append(xma.text)
        for ymi in root.iter('ymin'):
            ymin.append(ymi.text)
        for yma in root.iter('ymax'):
            ymax.append(yma.text)
        bbox_gt[xfile] = np.array([xmin,ymin,xmax,ymax], dtype=np.int).transpose().tolist()
        
    return bbox_gt

if __name__ == "__main__":
    
    files = glob.glob("*.xml")
    bbox_gt = parseXMLfiles(files)

    parser = argparse.ArgumentParser()

    parser.add_argument('-b', '--bbox_file', type=str)
    args = parser.parse_args()

    bbox_file = args.bbox_file
    bbox_pred = parseListsPred(bbox_file)
    
    sorted_bbox_pred, sorted_bbox_gt = sortLists(bbox_pred, bbox_gt)

    calculateTPFPdict(sorted_bbox_pred, sorted_bbox_gt)

    ##display predicted images + gt's with bboxes
    ##find the avarage power calculated for preds and gt

    #displayImages(sorted_bbox_pred, sorted_bbox_gt)
    #avgPWR(sorted_bbox_pred, sorted_bbox_gt)

    #print(bbox_pred)
    #print(bbox_gt)

    #for keys_pred,keys_gt  in zip(sorted_bbox_pred.keys(), sorted_bbox_gt.keys()):
    #    print(keys_pred[:-4],keys_gt[:-4])
