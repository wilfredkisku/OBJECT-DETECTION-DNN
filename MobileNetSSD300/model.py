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
        #print(state_dict['conv7.weight'].shape)
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
        
        conv_fc6_weight = pretrained_state_dict['classifier.0.weight'].view(4096, 512, 7, 7)
        conv_fc6_bias = pretrained_state_dict['classifier.0.bias']

        state_dict['conv6.weight'] = decimate(conv_fc6_weight, m=[4, None, 3, 3])
        state_dict['conv6.bias'] = decimate(conv_fc6_bias, m=[4])
    
        conv_fc7_weight = pretrained_state_dict['classifier.3.weight'].view(4096, 4096, 1, 1)
        conv_fc7_bias = pretrained_state_dict['classifier.3.bias']

        state_dict['conv7.weight'] = decimate(conv_fc7_weight, m=[4, 4, None, None])
        state_dict['conv7.bias'] = decimate(conv_fc7_bias, m=[4])

        self.load_state_dict(state_dict)
        #print(state_dict['conv7.weight'].shape)
        print('Loaded base model ...')

class AuxiliaryConvolutions(nn.Module):
    def __init__(self):
        super(AuxiliaryConvolutions, self).__init__()

        self.conv8_1 = nn.Conv2d(1024, 256, kernel_size=1, padding=0) #stride=1 
        self.conv8_2 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)

        self.conv9_1 = nn.Conv2d(512, 128, kernel_size=1, padding=0)
        self.conv9_2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)

        self.conv10_1 = nn.Conv2d(256, 128, kernel_size=1, padding=0)
        self.conv10_2 = nn.Conv2d(128, 256, kernel_size=3, padding=0)

        self.conv11_1 = nn.Conv2d(256, 128, kernel_size=1, padding=0)
        self.conv11_2 = nn.Conv2d(128, 256, kernel_size=3, padding=0)

        self.init_conv2d()

    def init_conv2d(self):
        for c in self.children():
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_uniform_(c.weight)
                nn.init.constant_(c.bias, 0.)

    def forward(self, conv7_feats):
        
        out = F.relu(self.conv8_1(conv7_feats))     # (N, 256, 19, 19)
        out = F.relu(self.conv8_2(out))             # (N, 512, 10, 10)
        conv8_2_feats = out                         # (N, 512, 10, 10)

        out = F.relu(self.conv9_1(out))             # (N, 128, 10, 10)
        out = F.relu(self.conv9_2(out))             # (N, 256, 5, 5)
        conv9_2_feats = out                         # (N, 256, 5, 5)

        out = F.relu(self.conv10_1(out))            # (N, 128, 5, 5)
        out = F.relu(self.conv10_2(out))            # (N, 256, 3, 3)
        conv10_2_feats = out                        # (N, 256, 3, 3)

        out = F.relu(self.conv11_1(out))            # (N, 128, 3, 3)
        conv11_2_feats = F.relu(self.conv11_2(out)) # (N, 256, 1, 1)

        return conv8_2_feats, conv9_2_feats, conv10_2_feats, conv11_2_feats

class PredictionConvolutions(nn.Module):

    def __init__(self, n_classes):

        super(PredictionConvolutions, self).__init__()

        self.n_classes = n_classes

        n_boxes = {'conv4_3':4,
                    'conv7':6,
                    'conv8_2':6,
                    'conv9_2':6,
                    'conv10_2':4,
                    'conv11_2':4
                }
        #localization prediction convolutions 
        self.loc_conv4_3 = nn.Conv2d(512, n_boxes['conv4_3']*4, kernel_size=3, padding=1)
        self.loc_conv7 = nn.Conv2d(1024, n_boxes['conv7']*4, kernel_size=3, padding=1)
        self.loc_conv8_2 = nn.Conv2d(512, n_boxes['conv8_2']*4, kernel_size=3, padding=1)
        self.loc_conv9_2 = nn.Conv2d(256, n_boxes['conv9_2']*4, kernel_size=3, padding=1)
        self.loc_conv10_2 = nn.Conv2d(256, n_boxes['conv10_2']*4, kernel_size=3, padding=1)
        self.loc_conv11_2 = nn.Conv2d(256, n_boxes['conv11_2']*4, kernel_size=3, padding=1)

        #class prediction convolutions
        self.cl_conv4_3 = nn.Conv2d(512, n_boxes['conv4_3']*n_classes, kernel_size=3, padding=1)
        self.cl_conv7 = nn.Conv2d(1024, n_boxes['conv7']*n_classes, kernel_size=3, padding=1)
        self.cl_conv8_2 = nn.Conv2d(512, n_boxes['conv8_2']*n_classes, kernel_size=3, padding=1)
        self.cl_conv9_2 = nn.Conv2d(256, n_boxes['conv9_2']*n_classes, kernel_size=3, padding=1)
        self.cl_conv10_2 = nn.Conv2d(256, n_boxes['conv10_2']*n_classes, kernel_size=3, padding=1)
        self.cl_conv11_2 = nn.Conv2d(256, n_boxes['conv11_2']*n_classes, kernel_size=3, padding=1)

        self.init_conv2d()

    def init_conv2d(self):
        for c in self.children():
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_uniform_(c.weight)
                nn.init.constant_(c.bias, 0.)

    def forward(self, conv4_3_feats, conv7_feats, conv8_2_feats, con9v_2_feats, conv10_2_feats, conv11_2_feats):

        batch_size = conv4_3_feats.size(0)
        
        l_conv4_3 = self.loc_conv4_3(conv4_3_feats)
        l_conv4_3 = l_conv4_3.permute(0, 2, 3, 1).contiguous()
        l_conv4_3 = l_conv4_3.view(batch_size, -1, 4) 

        l_conv7 = self.loc_conv7(conv7_feats)
        l_conv7 = l_conv7.permute(0, 2, 3, 1).contiguous()
        l_conv7 = l_conv7.view(batch_size, -1, 4)

        l_conv8_2 = self.loc_conv8_2(conv8_2_feats)
        l_conv8_2 = l_conv8_2.permute(0, 2, 3, 1).contiguous()
        l_conv8_2 = l_conv8_2.view(batch_size, -1, 4)

        l_conv9_2 = self.loc_conv9_2(conv9_2_feats)
        l_conv9_2 = l_conv9_2.permute(0, 2, 3, 1).contiguous()
        l_conv9_2 = l_conv9_2.view(batch_size, -1, 4)

        l_conv10_2 = self.loc_conv10_2(conv10_2_feats)
        l_conv10_2 = l_conv10_2.permute(0, 2, 3, 1).contiguous()
        l_conv10_2 = l_conv10_2.view(batch_size, -1, 4)

        l_conv11_2 = self.loc_conv11_2(conv11_2_feats)
        l_conv11_2 = l_conv11_2.permute(0, 2, 3, 1).contiguous()
        l_conv11_2 = l_conv11_2.view(batch_size, -1, 4)

        #predict classes
        c_conv4_3 = self.cl_conv4_3(conv4_3_feats)
        c_conv4_3 = c_conv4_3.permute(0, 2, 3, 1).contiguous()
        c_conv4_3 = c_conv4_3.view(batch_size, -1, self.n_classes)

        c_conv7 = self.cl_conv7(conv7_feats)
        c_conv7 = c_conv7.permute(0, 2, 3, 1).contiguous()
        c_conv7 = c_conv7.view(batch_size, -1, self.n_classes)

        c_conv8_2 = self.cl_conv8_2(conv8_2_feats)
        c_conv8_2 = c_conv8_2.permute(0, 2, 3, 1).contiguous()
        c_conv8_2 = c_conv8_2.view(batch_size, -1, self.n_classes)

        c_conv9_2 = self.cl_conv9_2(conv9_2_feats)
        c_conv9_2 = c_conv9_2.permute(0, 2, 3, 1).contiguous()
        c_conv9_2 = c_conv9_2.view(batch_size, -1, self.n_classes)

        c_conv10_2 = self.cl_conv10_2(conv10_2_feats)
        c_conv10_2 = c_conv10_2.permute(0, 2, 3, 1).contiguous()
        c_conv10_2 = c_conv10_2.view(batch_size, -1, self.n_classes)

        c_conv11_2 = self.cl_conv11_2(conv11_2_feats)
        c_conv11_2 = c_conv11_2.permute(0, 2, 3, 1).contiguous()
        c_conv11_2 = c_conv11_2.view(batch_size, -1, self.n_classes)

        locs = torch.cat([l_conv4_3, l_conv7, l_conv8_2, l_conv9_2, l_conv10_2, l_conv11_2], dim=1)
        classes_scores = torch.cat([c_conv4_3, c_conv7, c_conv8_2, c_conv9_2, c_conv10_2, c_conv11_2], dim=1)
        
        return locs, classes_scores

class SSD300(nn.Module):

    def __init__(self, n_classes):

        super(SSD300, self).__init__()
        
        self.n_classes = n_classes

        self.base = VGGBase()
        self.aux_convs = AuxiliaryConvolutions()
        self.pred_convs = PredictionConvolutions(n_classes)

        self.rescale_factors = nn.Parameter(torch.FloatTensor(1, 512, 1, 1))
        nn.init.constant_(self.rescale_factors, 20)

        self.priors_cxcy = self.create_prior_boxes()
        
    def forward(self, image):

        conv4_3_feats, conv7_feats = self.base(image)

        norm = conv4_3_feats.pow(2).sum(dim=1, keepdim=True).sqrt()
        conv4_3_feats = conv4_3_feats / norm
        conv4_3_feats = conv4_3_feats * self.rescale_factors
        
        conv8_2_feats, conv9_2_feats, conv10_2_feats, conv11_2_feats = self.aux_convs(conv7_feats)

        locs, classes_scores = self.pred_convs(conv4_3_feats, conv7_feats, conv8_2_feats, conv9_2_feats, conv10_2_feats, conv11_2_feats)

        return locs, classes_scores

    def create_prior_boxes(self):
        return None

if __name__ == "__main__":

    #model = VGGBase()
    #model = AuxiliaryConvolutions()
    #model = PredictionConvolutions(10)
    model = SSD300(10)
