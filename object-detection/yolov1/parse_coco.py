import os
import yaml
import numpy as np
import pandas as pd

import torch

coco_path = '/home/wilfred/dataset/COCO/archive/coco2017'
train_folder = 'train2017'
test_folder = 'test2017'
val_folder = 'val2017'

annotations_folder = 'annotations'

from collections import defaultdict
import json
import numpy as np

class COCOParser:
    def __init__(self, anns_file, imgs_dir=""):
        # format -> {image_id : [[id, category_id, bbox],[width, height]]}
        # bbox -> [x, y, width, height]
        with open(anns_file, 'r') as f:
            coco = json.load(f)

        self.annIm_dict = defaultdict(list)
        self.cat_dict = {}
        self.annId_dict = {}
        self.im_dict = {}
        self.licenses_dict = {}

        for ann in coco['annotations']:
            self.annIm_dict[ann['image_id']].append(ann)
            self.annId_dict[ann['id']]=ann
        for img in coco['images']:
            self.im_dict[img['id']] = img
        for cat in coco['categories']:
            self.cat_dict[cat['id']] = cat
        for license in coco['licenses']:
            self.licenses_dict[license['id']] = license

        self.imgId_class_data = defaultdict(list)

        for ann in coco['annotations']:
            self.imgId_class_data[ann['image_id']].append([ann['id'], ann['category_id'], ann['bbox']])
        
        #print(len(self.imgId_class_data))

        for info in coco['images']:
            self.imgId_class_data[info['id']].append([info['width'], info['height'], info['file_name']])

        #print(len(self.imgId_class_data))
        
        for idx in self.imgId_class_data:
            lst = self.imgId_class_data[idx][:-1]
            w, h, fname = self.imgId_class_data[idx][-1][0], self.imgId_class_data[idx][-1][1], self.imgId_class_data[idx][-1][2]
            '''
            for l in lst:
                if l[1] == 1:
                    print(idx, fname, (l[2][0]+(l[2][2]/2))/w, (l[2][1]+(l[2][3]/2))/h, l[2][2]/w, l[2][3]/h)
            print()
            '''
            #print(len(lst))
            leng = len([x for x in lst if x[1] == 1])
            if leng > 0:
                myfile = open('coco_dataset/train/'+fname[:-4]+'.txt', 'w')
                for l in lst:
                    #var1, var2 = line.split(",");
                    if l[1] == 1:
                        myfile.write(str(0)+' '+str((l[2][0]+(l[2][2]/2))/w)+' '+str((l[2][1]+(l[2][3]/2))/h)+' '+str(l[2][2]/w)+' '+str(l[2][3]/h)+'\n')

                myfile.close()
                #text_file.close()

    '''
    def get_imgIds(self):
        return list(self.im_dict.keys())
    
    def get_annIds(self, im_ids):
        im_ids=im_ids if isinstance(im_ids, list) else [im_ids]
        return [ann['id'] for im_id in im_ids for ann in self.annIm_dict[im_id]]
    
    def load_anns(self, ann_ids):
        im_ids=ann_ids if isinstance(ann_ids, list) else [ann_ids]
        return [self.annId_dict[ann_id] for ann_id in ann_ids]
    
    def load_cats(self, class_ids):
        class_ids=class_ids if isinstance(class_ids, list) else [class_ids]
        return [self.cat_dict[class_id] for class_id in class_ids]
    
    def get_imgLicenses(self,im_ids):
        im_ids=im_ids if isinstance(im_ids, list) else [im_ids]
        lic_ids = [self.im_dict[im_id]["license"] for im_id in im_ids]
        return [self.licenses_dict[lic_id] for lic_id in lic_ids]
    '''

def main():
    ann_path = os.path.join(coco_path, train_folder)
    lst = sorted(os.listdir(os.path.join(coco_path, annotations_folder)))
    
    COCOParser(os.path.join(coco_path, annotations_folder, lst[2]))
    
if __name__ == "__main__":
    main()
