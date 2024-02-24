import albumentations as A
import cv2
import torch

from albumentations.pytorch import ToTensorV2
from utils import seed_everything

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 2
BATCH_SIZE = 16
IMAGE_SIZE = 416
NUM_CLASSES = 1
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 1000
CONF_THRESHOLD = 0.05
MAP_IOU_THRESH = 0.5
NMS_IOU_THRESH = 0.45
S = [IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8]
PIN_MEMORY = True
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_FILE = "/home/wilfred/Desktop/object-detection/YOLOV3/training/checkpoint.pth.tar"

IMG_DIR = "/home/wilfred/Desktop/object-detection/YOLOV3/data/TrainImageFolder/images"
LABEL_DIR = "/home/wilfred/Desktop/object-detection/YOLOV3/data/train_labels_persons"

IMG_DIR_TEST = "/home/wilfred/Desktop/object-detection/YOLOV3/data/ValImageFolder/images"
LABEL_DIR_TEST = "/home/wilfred/Desktop/object-detection/YOLOV3/data/val_labels_persons"


ANCHORS = [[(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)], [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)], [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],]

scale = 1.1

train_transforms = A.Compose(
    [   
        #Rescale an image so that maximum side is equal to max_size, keeping the aspect ratio of the initial image.
        A.LongestMaxSize(max_size=int(IMAGE_SIZE * scale)),
        #Pad side of the image / max if side is less than desired number.
        A.PadIfNeeded(min_height=int(IMAGE_SIZE * scale), min_width=int(IMAGE_SIZE * scale), border_mode=cv2.BORDER_CONSTANT,),
        #take an input image, extract a random patch with size height=IMAGE_SIZE by width=IMAGE_SIZE pixels
        A.RandomCrop(width=IMAGE_SIZE, height=IMAGE_SIZE),
        #Randomly changes the brightness, contrast, and saturation of an image.
        A.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.6, p=0.4),
        #the OneOfblock applies one of the augmentations inside it. That means that if the random generator chooses to apply OneOf then one child augmentation from it will be applied to the input data.
        #A.OneOf([A.ShiftScaleRotate(rotate_limit=20, p=0.5, border_mode=cv2.BORDER_CONSTANT),A.Affine(shear=15, p=0.5, mode="constant")], p=1.0,),
        #Flip the input horizontally around the y-axis.
        A.HorizontalFlip(p=0.5),
        #Blur the input image using a random-sized kernel.
        A.Blur(p=0.1),
        #Apply Contrast Limited Adaptive Histogram Equalization to the input image.
        A.CLAHE(p=0.1),
        #Reduce the number of bits for each color channel.
        A.Posterize(p=0.1),
        #Convert the input RGB image to grayscale. If the mean pixel value for the resulting image is greater than 127, invert the resulting grayscale image.
        A.ToGray(p=0.1),
        #Randomly rearrange channels of the input RGB image.
        A.ChannelShuffle(p=0.05),
        #Normalization is applied by the formula: img = (img - mean * max_pixel_value) / (std * max_pixel_value)
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255,),
        #Converts images/masks to PyTorch Tensors, inheriting from BasicTransform. Supports images in numpy HWC format and converts them to PyTorch CHW format. If the image is in HW format, it will be converted to PyTorch HW.
        ToTensorV2(),
    ],
    bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[],),
)
test_transforms = A.Compose(
    [
        A.LongestMaxSize(max_size=IMAGE_SIZE),
        A.PadIfNeeded(min_height=IMAGE_SIZE, min_width=IMAGE_SIZE, border_mode=cv2.BORDER_CONSTANT),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255,),
        ToTensorV2(),
    ],
    bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[]),
)

PASCAL_CLASSES = ["person"]
COCO_LABELS = ["person"]
