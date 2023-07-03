from PIL import Image, ImageDraw

import re
import numpy as np
import matplotlib.pyplot as plt
import cv2

import xml.etree.ElementTree as ET


def read_content(xml_file: str):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    list_with_all_boxes = []

    for boxes in root.iter('object'):
        
        print(boxes)

        filename = root.find('filename').text

        ymin, xmin, ymax, xmax = None, None, None, None

        ymin = int(boxes.find("bndbox/ymin").text)
        xmin = int(boxes.find("bndbox/xmin").text)
        ymax = int(boxes.find("bndbox/ymax").text)
        xmax = int(boxes.find("bndbox/xmax").text)

        list_with_single_boxes = [xmin, ymin, xmax, ymax]
        print(list_with_single_boxes)
        list_with_all_boxes.append(list_with_single_boxes)

    return filename, list_with_all_boxes

def drawImages():
    DATA_PATH = '/home/wilfred/dataset/StanfordDogDataset/archive/'

    with open(DATA_PATH+'annotations/Annotation/n02113799-standard_poodle/n02113799_489') as f:
        reader = f.read()

    img = Image.open(DATA_PATH+'images/Images/n02113799-standard_poodle/n02113799_489.jpg')
    img_= cv2.imread(DATA_PATH+'images/Images/n02113799-standard_poodle/n02113799_489.jpg')

    print(np.asarray(img))
    print(img_[:,:,::-1])

    plt.imshow(img_[:,:,::-1])
    plt.show()

    xmin = int(re.findall('(?<=<xmin>)[0-9]+?(?=</xmin>)', reader)[0])
    xmax = int(re.findall('(?<=<xmax>)[0-9]+?(?=</xmax>)', reader)[0])
    ymin = int(re.findall('(?<=<ymin>)[0-9]+?(?=</ymin>)', reader)[0])
    ymax = int(re.findall('(?<=<ymax>)[0-9]+?(?=</ymax>)', reader)[0])

    print(r"xmin={}, ymin={}, xmax={}, ymax={}".format(xmin, ymin, xmax, ymax))
    
    #origin_img = img.copy()
    #draw = ImageDraw.Draw(origin_img)
    #draw.rectangle(xy=[(xmin,ymin), (xmax,ymax)])

if __name__ == '__main__':
    DATA_PATH = '/home/wilfred/dataset/StanfordDogDataset/archive/'
    path = '/home/wilfred/dataset/PASCAL-VOC/archive/VOC2012_train_val/VOC2012_train_val/Annotations/2007_000027.xml'
    #drawImages()
    #name, boxes = read_content(DATA_PATH+'annotations/Annotation/n02113799-standard_poodle/n02113799_489')
    name, boxes = read_content(path)
    print(name, boxes)
