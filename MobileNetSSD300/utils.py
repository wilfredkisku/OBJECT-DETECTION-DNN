from PIL import Image, ImageDraw

import re
import numpy as np
import matplotlib.pyplot as plt
import cv2


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

    drawImages()


