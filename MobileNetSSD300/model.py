from torch import nn
from utils import *
from math import sqrt
from itertools import product as product

import torchvision
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VGGBase(nn.Module):
    """
    VGG base convolutions to produce lower-level feature maps.
    """

    def _init__(self):
        super(VGGBase, self).__init__()

        #standard convolution layers in VGG16
        
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1) #stride=1 by default
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True) #ceiling (not floor) for even dimenions

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1) #retain size with stride as 1 (and padding)

        self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6) #atrous convolution
        self.conv7 = nn.Conv2d(1024, 1024, kernel_size=1)

        #Load the pretrained layers of VGG16 onto the newly created state dictionary (Imagenet)

        self.load_pretrained_layers()

    def forward(self, image):
        """
        Forward Propagation

        Param:
            image: images, a tensor of dimension (N, 3, 300, 300)
        Returns:
            conv4_3_feats, conv7: lower-level feature maps
        """

        out = F, relu(self.conv1_1(image)) # => (N, 64, 300, 300)
        out = F.relu(self.conv1_2(out)) # => (N, 64, 300, 300)
        out = self.pool1(out) # => (N, 64, 150, 150)

        out = F, relu(self.conv2_1(out)) # => (N, 128, 150, 150)
        out = F.relu(self.conv2_2(out)) # => (N, 128, 150, 150)
        out = self.pool2(out) # => (N, 128, 75, 75)

        out = F.relu(self)

        return conv4_3_feats, conv7_feats

if __name__ == "__main__":

    model = VGGBase()
