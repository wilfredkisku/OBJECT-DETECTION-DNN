import os
import ast
import cv2
import json
import torch
import numpy as np

from torch.utils.data import Dataset
from PIL import Image
from utils import transform
import xml.etree.ElementTree as ET

class VOCDataset(Dataset):
    def __init__(self, DataFolder, split):
        
        self.split = str(split.upper())
        
        if self.split not in {"TRAIN", "TEST"}:
            print("Param split not in {TRAIN, TEST}")
            assert self.split in {"TRAIN", "TEST"}

        self.DataFolder = DataFolder

        #read data file from json file
        ## READ WITH AST
        with open(os.path.join(DataFolder, self.split+ '_images.json'), 'r') as j:
            #self.images = json.load(j)
            self.images = ast.literal_eval(j.read())

        with open(os.path.join(DataFolder, self.split+ '_objects.json'), 'r') as j:
            #self.objects = json.load(j)
            self.objects = ast.literal_eval(j.read())

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


def read_json(path_img, path_anno, path_coco, data="COCO"):
    
    if data == "PASCAL": 

        anno_f = os.listdir(path_anno)
        
        for f in anno_f:
            file_path = os.path.join(path_anno,f)
            img_path = os.path.join(path_img,f[:-3]+"jpg")
            try:
                #PARSE THE XML FILE
                tree = ET.parse(file_path)
                root = tree.getroot()

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

        ##READ THE COCO-ANNOTATIONS
        f = open(path_coco)
        data = json.load(f)
        
        '''
        returns --> [{image_id:..., bbox:[[]...], labels:[...], difficulties:[...]}]
        count --> 4952
        '''
        
        ##INFO EXTRACTION
        dict_list = []
        unique_img_id = []
        
        count = 0

        for d in data['annotations']:
            
            #DEBUG LINE
            #print(d['image_id'], d['bbox'], d['category_id'])
            
            dict_ = {}
            
            if d['image_id'] not in unique_img_id:
                unique_img_id.append(d['image_id'])
                dict_['image_id'] = d['image_id']
                dict_['bbox'] = [d['bbox']]
                dict_['labels'] = [d['category_id']]
                dict_list.append(dict_)
            else:
                for item in dict_list:
                    if item['image_id'] == d['image_id']:
                        item['bbox'].append(d['bbox'])
                        item['labels'].append(d['category_id'])

            count += 1
        
        print(dict_list)
        print("DICT COUNT :: "+str(len(dict_list)))
        print("COUNT UNIQUE :: "+str(len(unique_img_id)))

    return None

def text_parse(data_path, anno_path):
    
    ##LIST OF IMAGE FILES AND ANNOTATION FILES
    d_files = os.listdir(data_path)
    a_files = os.listdir(anno_path)

    #img_path = os.path.join(data_path, d_files[0])
    #img = cv2.imread(img_path)
    
    '''[cx, cy, w, h], [height, width, channels]'''
    
    ##EXTRACT FILES AND ANNOTATION LINES
    data_con = []
    path_con = []
    for d_f in d_files:
        with open(os.path.join(anno_path, d_f[:-3]+'txt')) as file:
            lines = [line.rstrip().split() for line in file]
        
        path_con.append(os.path.join(data_path, d_f))
        new_lines = []
        new_lines_dict = {key: [] for key in ["boxes", "labels", "difficulties"]}
        
        for l in lines:
            
            line = [float(li) for _, li in enumerate(l)]
            
            new_lines.append(line)
            
            img_path = os.path.join(data_path, d_f)
            img = cv2.imread(img_path)
            
            h = img.shape[0]
            w = img.shape[1]

            # x--> width, y--> height
            x1 = line[1] - line[3] / 2
            y1 = line[2] - line[4] / 2

            x2 = line[1] + line[3] / 2
            y2 = line[2] + line[4] / 2
            
            #print(int(w*x1), int(h*y1), int(w*x2), int(h*y2), ((x2-x1)*(y2-y1)*100))
            if (x2-x1)*(y2-y1)*100 < 1.0:
                #print('Difficult..')
                #cv2.rectangle(img, (int(w*x1), int(h*y1)), (int(w*x2), int(h*y2)), color=(0,255,0), thickness=2)
                #cv2.imshow("image", img)
                #cv2.waitKey(0)
                #cv2.destroyAllWindows()
                difficulty = 1
            else:
                #print('Easy..')
                difficulty = 0
            
            new_lines_dict["boxes"].append([int(w*x1), int(h*y1), int(w*x2), int(h*y2)])
            new_lines_dict["labels"].append(1)
            new_lines_dict["difficulties"].append(difficulty)

        data_con.append(new_lines_dict)
    
    #print(data_con)
    with open('TRAIN_objects.json', 'w') as f:
        f.write(str(data_con))

    with open('TRAIN_images.json', 'w') as f:
        f.write(str(path_con))
    #print(os.path.join(data_path, d_f),len(new_lines))

    '''
    #verified
    h = img.shape[0]
    w = img.shape[1]
    
    # x--> width, y--> height
    x1 = new_lines[0][1] - new_lines[0][3] / 2
    y1 = new_lines[0][2] - new_lines[0][4] / 2

    x2 = new_lines[0][1] + new_lines[0][3] / 2
    y2 = new_lines[0][2] + new_lines[0][4] / 2

    cv2.rectangle(img, (int(w*x1), int(h*y1)), (int(w*x2), int(h*y2)), color=(0,255,0), thickness=2)
    
    cv2.imshow("image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''

    return None

if __name__ == "__main__":

    #coco_path = "/home/wilfred/dataset/COCO/archive/coco2017/annotations/instances_val2017.json"
    #json_path = "/home/wilfred/dataset/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/Annotations"
    #img_path = "/home/wilfred/dataset/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/JPEGImages"
    #read_json(img_path, json_path, coco_path, "COCO")

    #data_path = "/home/wilfred/Desktop/object-detection/yolov1/data/TrainImageFolder/images"
    #anno_path = "/home/wilfred/Desktop/object-detection/yolov1/data/train_labels_persons"

    #text_parse(data_path, anno_path)

    from utils import combine

    train_dataset = VOCDataset('./', split= "train")
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size= 8, shuffle= True, collate_fn= combine, pin_memory = True)
    data = next(iter(train_loader))
    print(data[2].shape)
