import os
#os.environ['MPLCONFIGDIR'] = '/home/wilfred/Dataset/dgx/object-detection/yolov1/tmp/'
import sys
import json
from pathlib import Path

import cv2
import yaml
import torch
import numpy as np
from tqdm import tqdm

from transform import BasicTransform, AugmentTransform
from utils import transform_xcycwh_to_x1y1wh

### label --> txt file --> [label, x_c, y_c, width_obj, height_obj]
### image --> rescaled --> (128, 128)

class Dataset:
    #INIT CONFIGURATION
    def __init__(self, yaml_path, phase="train"):
        
        with open(yaml_path, mode="r") as f:
            data_item = yaml.load(f, Loader=yaml.FullLoader)

        self.phase = phase
        self.class_list = data_item["CLASS_INFO"]
        
        self.image_paths = []

        for sub_dir in data_item[self.phase.upper()]:
            image_dir = Path(data_item["PATH"]) / sub_dir
            self.image_paths += [str(image_dir / fn) for fn in os.listdir(image_dir) if fn.lower().endswith(("png", "jpg", "jpeg"))]

        self.label_paths = self.replace_image2label_path(data_item, self.image_paths)
        self.generate_no_label(self.label_paths)

        self.mAP_filepath = None
        if phase == "val":
            self.generate_mAP_source(save_dir=Path("./data/eval_src"), mAP_filename=data_item["VAL_FILE"])

    #LENGTH
    def __len__(self):
        return len(self.image_paths)

    #GETITEM
    def __getitem__(self, index):
        filename, image, label = self.get_GT_item(index)
        shape = image.shape[:2]
        image, boxes, labels = self.transformer(image=image, boxes=label[:,1:5], labels=label[:,0])
        img_tensor = self.to_tensor(image)
        label = torch.from_numpy(np.concatenate((labels[:, np.newaxis], boxes), axis=1))
        return filename, img_tensor, label, shape

    def get_GT_item(self, index):
        filename, image = self.get_image(index)
        label = self.get_label(index)
        label = self.check_no_label(label)
        return filename, image, label

    def to_tensor(self, image):
        image = np.ascontiguousarray(image)
        return torch.from_numpy(image).float()

    def get_image(self, index):
        filename = self.image_paths[index].split(os.sep)[-1]
        image = cv2.imread(self.image_paths[index])
        #BGR TO RGB CONVERSION
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return filename, image


    def get_label(self, index):
        with open(self.label_paths[index], mode="r") as f:
            item = [x.split() for x in f.read().splitlines()]
        return np.array(item, dtype=np.float32)

    #create labels from image paths
    def replace_image2label_path(self, data_item, image_paths):
        #sa, sb = f"{os.sep}images{os.sep}", f"{os.sep}labels_persons_new{os.sep}"
        #return [sb.join(x.rsplit(sa, 1)).rsplit(".", 1)[0] + ".txt" for x in image_paths]
        return [str(Path(data_item["PATH"]) / data_item[(self.phase + "_labels_persons").upper()][0]) + "/" + x.split('/')[-1][:-4] + ".txt" for x in image_paths]

    #fill in files for no images (empty files for no class images)
    def generate_no_label(self, label_paths):
        for label_path in label_paths:
            if not os.path.isfile(label_path):
                f = open(str(label_path), mode="w")
                f.close()

    def check_no_label(self, label):
        if len(label) == 0:
            label = np.array([[-1, 0, 0, 0, 0]], dtype=np.float32)
        return label


    def load_transformer(self, transformer):
        self.transformer = transformer
    
    #for validation
    def generate_mAP_source(self, save_dir, mAP_filename):
        
        #create a directory if there is none
        if not save_dir.is_dir():
            os.makedirs(save_dir, exist_ok=True)
        
        #save it into the mAP filepath variable
        self.mAP_filepath = save_dir / mAP_filename
        
        #create a file file with the metrics
        if not self.mAP_filepath.is_file():
            
            #classlist --> {0} --> person class
            class_id2category = self.class_list
            
            #formatter details
            mAP_file_formatter = {}
            mAP_file_formatter["imageToid"] = {}
            mAP_file_formatter["images"] = []
            mAP_file_formatter["annotations"] = []
            mAP_file_formatter["categories"] = []

            #generating unique ids for each images datapoint
            lbl_id = 0

            for i in tqdm(range(len(self))):

                #get file details from the ground truth
                filename, image, label = self.get_GT_item(i)
                #image shape
                img_h, img_w = image.shape[:2]
                
                mAP_file_formatter["imageToid"][filename] = i
                mAP_file_formatter["images"].append({"id": i, "width": img_w, "height": img_h})
                
                label[:, 1:5] = transform_xcycwh_to_x1y1wh(label[:, 1:5])
                label[:, [1,3]] *= img_w
                label[:, [2,4]] *= img_h
                
                for j in range(len(label)):
                    x = {}
                    x["id"] = lbl_id
                    x["image_id"] = i
                    x["bbox"] = [round(item, 2) for item in label[j][1:5].tolist()]
                    x["area"] = round((x["bbox"][2] * x["bbox"][3]), 2)
                    x["iscrowd"] = 0
                    x["category_id"] = int(label[j][0])
                    x["segmentation"] = []
                    mAP_file_formatter["annotations"].append(x)
                    lbl_id += 1

            for i, cate_name in class_id2category.items():
                mAP_file_formatter["categories"].append({"id": i, "supercategory": "", "name": cate_name})

            with open(self.mAP_filepath, "w") as outfile:
                json.dump(mAP_file_formatter, outfile)

    @staticmethod
    def collate_fn(minibatch):
        filenames = []
        images = []
        labels = []
        shapes = []
        
        for _, items in enumerate(minibatch):
            filenames.append(items[0])
            images.append(items[1])
            labels.append(items[2])
            shapes.append(items[3])
        return filenames, torch.stack(images, dim=0), labels, shapes


if __name__ == "__main__":

    yaml_path = "voc_person.yaml"
    input_size = 128

    train_dataset = Dataset(yaml_path=yaml_path, phase='train')
    train_transformer = AugmentTransform(input_size=input_size)
    train_dataset.load_transformer(transformer=train_transformer)

    val_dataset = Dataset(yaml_path=yaml_path, phase='val')
    val_transformer = BasicTransform(input_size=input_size)
    val_dataset.load_transformer(transformer=val_transformer)

    print(len(train_dataset), len(val_dataset))

    for data in train_dataset:
        print(data[1].shape)
        break
