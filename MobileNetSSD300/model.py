import torch 

from torch import nn
from utils import *
from math import sqrt
from itertools import product as product

import torchvision
import torch.nn.functional as F

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(device)

class VGGBase(nn.Module):
    """
    VGG base convolutions to produce lower-level feature maps.
    """

    def __init__(self):
        super(VGGBase, self).__init__()

        #standard convolution layers in VGG16
        # CONV 1_1 => CONV 1_2 => POOL  
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1) #stride=1 by default
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # CONV 2_1 => CONV 2_2 => POOL
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # CONV 3_1 => CONV 3_2 => CONV 3_3 => POOL
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True) #ceiling (not floor) for even dimenions

        # CONV 4_1 => CONV 4_2 => CONV 4_3 => POOL
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # CONV 5_1 => CONV 5_2 => CONV 5_3 => POOL
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

        out = F.relu(self.conv1_1(image)) # => (N, 64, 300, 300)
        out = F.relu(self.conv1_2(out)) # => (N, 64, 300, 300)
        out = self.pool1(out) # => (N, 64, 150, 150)

        out = F.relu(self.conv2_1(out)) # => (N, 128, 150, 150)
        out = F.relu(self.conv2_2(out)) # => (N, 128, 150, 150)
        out = self.pool2(out) # => (N, 128, 75, 75)

        out = F.relu(self.conv3_1(out)) # => (N, 256, 75, 75)
        out = F.relu(self.conv3_2(out)) # => (N, 256, 75, 75)
        out = F.relu(self.conv3_3(out)) # => (N, 256, 75, 75)
        out = self.pool3(out) # => (N, 256, 75/2 => 38, 75/2 => 38) with ceil_mode = True

        out = F.relu(self.conv4_1(out)) # => (N, 512, 38, 38)
        out = F.relu(self.conv4_2(out)) # => (N, 512, 38, 38)
        out = F.relu(self.conv4_3(out)) # => (N, 512, 38, 38)
        conv4_3_feats = out # => (N, 512, 38, 38)
        out = self.pool4(out) # => (N, 512, 19, 19)

        out = F.relu(self.conv5_1(out)) # => (N, 512, 19, 19)
        out = F.relu(self.conv5_2(out)) # => (N, 512, 19, 19)
        out = F.relu(self.conv5_3(out)) # => (N, 512, 19, 19)
        out = self.pool5(out) # => (N, 512, 19, 19)

        out = F.relu(self.conv6(out)) # => (N, 1024, 19, 19)

        conv7_feats = F.relu(self.conv7(out)) # => (N, 1024, 19, 19)

        return conv4_3_feats, conv7_feats

    def load_pretrained_layers(self):

        #state dictionary of the current model
        state_dict = self.state_dict()
        #names of the paramaters
        param_names = list(state_dict.keys())

        #pretrained VGG base
        pretrained_state_dict = torchvision.models.vgg16(pretrained=True).state_dict()
        pretrained_param_names = list(pretrained_state_dict.keys())

        #print(state_dict)
        #print(param_names)
        #print(param_names[:-4])

        # Transfer conv parameters from pretrained model to current model
        for i, param in enumerate(param_names[:-4]): # excluding conv6 and conv7 parameters
            state_dict[param] = pretrained_state_dict[pretrained_param_names[i]]

if __name__ == "__main__":

    model = VGGBase()
