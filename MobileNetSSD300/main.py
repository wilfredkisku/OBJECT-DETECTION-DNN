import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import torch
from torchvision import models
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F
import torchvision

from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

import os
import json
import re
import random
from math import sqrt

DATA_PATH = "/home/wilfred/dataset/StanfordDogDataset/archive"
print(os.listdir(DATA_PATH))
