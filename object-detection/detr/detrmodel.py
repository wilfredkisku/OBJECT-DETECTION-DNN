from PIL import Image
import requests
import matplotlib.pyplot as plt

import torch
from torch import nn
from torchvision.models import resnet50
import torchvision.transforms as T

torch.set_grad_enabled(False)

class DETRMODEL(nn.Module):
    # learned positional encoding instead of sine
    # positional encoding is passed at input (instead of attention)
    # fc bbox predictor instead of MLP

    def __init__(self, num_classes, hidden_dim = 256, nheads=8, num_encoder_layers=6, num_decoder_layers=6):
        super.__init__()

        #create the resnet backbone
        self.backbone = resnet50()
        del self.backbone.fc

        #create conversion layer
        self.conv = nn.Conv2d(2048, hidden_dim, 1)

        self.transformer = nn.Transformer(hidden_dim, nheads, num_encoder_layers, num_decoder_layers)
        self.linear_class = nn.Linear(hidde_dim, num_classes + 1)
        self.linear_bbox = nn.Linear(hidden_dim, 4)
