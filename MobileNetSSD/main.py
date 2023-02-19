import os
import glob
import argparse
import numpy as np

from utils import *


"""
    initFiles:
        assimilates the files in the test folder and also creates a list of lists for the datapoints
        [[dp1],[dp2],...] from the predicted txt file. It requires the bounidng box filename.
    Returns:
        (dict) bbox_gt | ground truths dictionary with filename as key and list of bounding box
        (dict) bbox_pred | predicted bbox infomration as a dictionary with filename keys and bbox ad list information
"""

def initFiles(bboxfile, xmlfiles):
    
    bbox_pred = parseListsPred(bboxfile)
    bbox_gt = parseXMLfiles(xmlfiles)
    

    #testing and sorting the bbox_gt
    myKeys_gt = list(bbox_gt.keys())
    myKeys_gt.sort()
    sorted_bbox_gt = {i: bbox_gt[i] for i in myKeys_gt}

    myKeys_pred = list(bbox_pred.keys())
    myKeys_pred.sort()
    sorted_bbox_pred = {i: bbox_pred[i] for i in myKeys_pred}

    return sorted_bbox_pred, sorted_bbox_gt

#test the method interfaces in main
if __name__ == "__main__":
    
    #command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--bbox_file', type=str) 
    args = parser.parse_args()

    #file for processing
    bboxfile = args.bbox_file
    xmlfiles = glob.glob("*.xml")

    #return the dicts into bbox_pred and bbox_gt
    sorted_bbox_pred, sorted_bbox_gt = initFiles(bboxfile, xmlfiles)

    #print(sorted_bbox_pred)
    #print(sorted_bbox_gt)

    for keys_pred,keys_gt  in zip(sorted_bbox_pred.keys(), sorted_bbox_gt.keys()):
        print(keys_pred[:-4] == keys_gt[:-4])
