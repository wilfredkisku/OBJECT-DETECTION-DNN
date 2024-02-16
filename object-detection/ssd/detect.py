from PIL import Image, ImageDraw, ImageFont
from utils import *
from torchvision import transforms
import torch
from model.SSD300 import SSD300
from model.vgg import VGG16BaseNet, AuxiliaryNet, PredictionNet


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

img_path = "./test/image.jpg"
trained_model = torch.load("model_state_ssd300.pth.tar")

