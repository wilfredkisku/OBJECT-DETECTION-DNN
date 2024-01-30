import torch
import torch.optim as optim

#All from model class
from model.SSD300 import SSD300
from model.vgg import VGG16BaseNet, AuxiliaryNet, PredictionNet
from model.multibox_loss import MultiBoxLoss
from model.metrics.metric import Metrics

from datasets import VOCDataset
from utils import *
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

