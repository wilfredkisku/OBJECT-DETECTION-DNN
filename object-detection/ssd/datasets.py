import torch
from torch.utils.data import Dataset
import json
import os
from PIL import Image
from utils import transform

class VOCDataset(Dataset):
    def __init__(self, DataFolder, split):
        
        self.split = str(split.upper())
        
        if self.split not in {"TRAIN", "TEST"}:
            print("Param split not in {TRAIN, TEST}")
            assert self.split in {"TRAIN", "TEST"}

        self.DataFolder = DataFolder

        #read data file from json file
        with open(os.path.join(DataFolder, self.split+ '_images.json'), 'r') as j:
            self.images = json.load(j)
        
        with open(os.path.join(DataFolder, self.split+ '_objects.json'), 'r') as j:
            self.objects = json.load(j)

        assert len(self.images) == len(self.objects)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):

        image = Image.open(self.images[i], mode= "r")
        image = image.convert("RGB")

        #Read objects in this image
        objects = self.objects[i]
        boxes = torch.FloatTensor(objects["boxes"])
        labels = torch.LongTensor(objects['labels'])
        difficulties = torch.ByteTensor(objects['difficulties'])

        #Apply transforms
        new_image, new_boxes, new_labels, new_difficulties = transform(image, boxes, labels, difficulties, self.split)

        return new_image, new_boxes, new_labels, new_difficulties

import os
import json
import xml.etree.ElementTree as ET

def read_json(path_img, path_anno, path_coco, data="COCO"):
    
    if data == "PASCAL": 
        #print(len(os.listdir(path_anno)))
        #print(len(os.listdir(path_img)))

        anno_f = os.listdir(path_anno)
        for f in anno_f:
            file_path = os.path.join(path_anno,f)
            img_path = os.path.join(path_img,f[:-3]+"jpg")
            try:
                # Parse the XML file
                tree = ET.parse(file_path)
                root = tree.getroot()

                #print(root.tag)
                
                #for child in root:
                #    print(child.tag, child.attrib)
                bboxes = []
                bbox = {}

                boxes = []
                labels = []
                difficulties = []

                for obj in root.findall('object'):
                    
                    #bbox['name'] = obj.find('name').text
                    #bbox['difficult'] = int(obj.find('difficult').text)
                    #bbox['xmin'] = int(obj.find('bndbox').find('xmin').text)
                    #bbox['ymin'] = int(obj.find('bndbox').find('ymin').text)
                    #bbox['xmax'] = int(obj.find('bndbox').find('xmax').text)
                    #bbox['ymax'] = int(obj.find('bndbox').find('ymax').text)
                    #bbox["boxes"] = [int(obj.find('bndbox').find('xmin').text), int(obj.find('bndbox').find('ymin').text), int(obj.find('bndbox').find('xmax').text), int(obj.find('bndbox').find('ymax').text)]
                    #bbox["labels"] = obj.find('name').text
                    #bbox["difficult"] = int(obj.find('difficult').text)
                    #if bbox['labels'] == 'person':
                    #bboxes.append(bbox)
                    
                    if obj.find('name').text == "person":
                        boxes.append([int(obj.find('bndbox').find('xmin').text), int(obj.find('bndbox').find('ymin').text), int(obj.find('bndbox').find('xmax').text), int(obj.find('bndbox').find('ymax').text)])
                        labels.append(obj.find('name').text)
                        difficulties.append(int(obj.find('difficult').text))
                        #print(boxes, labels, difficulties)
                
                if len(boxes) >= 1: 
                    bbox["path"] = img_path
                    bbox["boxes"] = boxes
                    bbox["labels"] = labels
                    bbox["difficulties"] = difficulties
                    bboxes.append(bbox)
                
                print(bboxes)
                #print(len(boxes), len(labels), len(difficulties))
                
                break
                '''
                # Print the root tag
                print("Root element:", root.tag)

                # Iterate through child elements
                for child in root:
                    print("Child element:", child.tag, "with attributes:", child.attrib)
                    
                    # Iterate through grandchildren if any
                    for grandchild in child:
                        print("\tGrandchild element:", grandchild.tag, "with text:", grandchild.text)
                '''
            except FileNotFoundError:
                print("File not found.")
            except ET.ParseError:
                print("Error parsing XML file.")
    else:

        # Opening JSON file
        f = open(path_coco)

        # returns JSON object as
        # a dictionary
        data = json.load(f)
        print(data['categories'])
        print(len(data['annotations']))        
    return None

if __name__ == "__main__":

    coco_path = "/home/wilfred/dataset/COCO/archive/coco2017/annotations/instances_val2017.json"
    json_path = "/home/wilfred/dataset/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/Annotations"
    img_path = "/home/wilfred/dataset/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/JPEGImages"
    read_json(img_path, json_path, coco_path, "COCO")
