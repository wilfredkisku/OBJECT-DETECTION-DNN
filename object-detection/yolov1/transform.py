import cv2
import torch
import numpy as np
from tqdm import tqdm
#from dataset import Dataset
from torchvision import datasets, transforms

MEAN = 0.4333
STD = 0.2194

################# calculate the mean and std ######################

transform_ms = transforms.Compose([transforms.Resize((128, 128)),
                                transforms.Grayscale(),
                                transforms.ToTensor(),])

def mean_and_std(path):

    dataset = datasets.ImageFolder(path, transform=transform_ms)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    nimages = 0
    mean = 0.
    std = 0.

    for batch, _ in tqdm(dataloader, desc="Progress"):
        # Rearrange batch to be the shape of [B, C, W * H]
        batch = batch.view(batch.size(0), batch.size(1), -1)
        # Update total number of images
        nimages += batch.size(0)
        # Compute mean and std here
        mean += batch.mean(2).sum(0) 
        std += batch.std(2).sum(0)
        
    # Final step
    mean /= nimages
    std /= nimages

    print(mean, std)

###################################################################

def to_tensor(image):
    image = np.ascontiguousarray(image.transpose(0,1))
    return torch.from_numpy(image).float()

def to_image(tensor, mean=MEAN, std=STD):
    denorm_tensor = tensor.clone()
    for t, m, s in zip(denorm_tensor, mean, std):
        t.mul_(s).add_(m)
    denorm_tensor.clamp_(min=0, max=1.)
    denorm_tensor *= 255
    #move the channel to the last --> CV2 representation
    image = denorm_tensor.permute(1,2,0).numpy().astype(np.uint8)
    return image

def denormalize(image, mean=MEAN, std=STD):
    image *= std
    image += mean
    image *= 255.
    return image.astype(np.uint8)

#### Transform Classes
class Compose:
    def __init__(self, tranforms):
        self.transforms = tranforms

    def __call__(self, image, boxes=None, labels=None):
        for tf in self.transforms:
            image, boxes, labels = tf(image, boxes, labels)

        return image, boxes, labels

class Normalize:
    def __init__(self, mean, std):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None):
        image /= 255.
        #image -= self.mean
        #image /= self.std
        return image, boxes, labels

class Resize:
    def __init__(self, size=128):
        self.size = size

    def __call__(self, image, boxes=None, labels=None):
        image = cv2.resize(image, (self.size, self.size))
        return image, boxes, labels

class GrayScale:
    def __call__(self, image, boxes=None, labels=None):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = np.reshape(image, (-1, image.shape[0], image.shape[1]))
        #print(image.shape)
        return image, boxes, labels

class ConcatenateChannels:
    def __call__(self, image, boxes=None, labels=None):
        image = np.concatenate((image, image, image), axis=0)
        return image, boxes, labels

class RandomBrightness:
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, image, boxes=None, labels=None):
        if np.random.randint(2):
            image += np.random.uniform(-self.delta, self.delta)
        return image, boxes, labels

class RandomContrast:
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image, boxes=None, labels=None):
        if np.random.randint(2):
            image *= np.random.uniform(self.lower, self.upper)
        return image, boxes, labels

### Dims transform
class ToAbsoluteCoords:
    def __call__(self, image, boxes, labels=None):
        height, width = image.shape
        boxes[:, [0, 2]] *= width
        boxes[:, [1, 3]] *= height
        return image, boxes, labels


class ToPercentCoords:
    def __call__(self, image, boxes, labels=None):
        height, width = image.shape
        boxes[:, [0, 2]] /= width
        boxes[:, [1, 3]] /= height
        return image, boxes, labels


class ToXminYminXmaxYmax:
    def __call__(self, image, boxes, labels=None):
        x1y1 = boxes[:, :2] - boxes[:, 2:] / 2
        x2y2 = boxes[:, :2] + boxes[:, 2:] / 2
        ##boxes are changed here why???
        boxes = np.concatenate((x1y1, x2y2), axis=1).clip(min=0, max=1)
        return image, boxes, labels


class ToXcenYcenWH:
    def __call__(self, image, boxes, labels=None):
        wh = boxes[:, 2:] - boxes[:, :2]
        xcyc = boxes[:, :2] + wh / 2
        boxes = np.concatenate((xcyc, wh), axis=1).clip(min=0, max=1)
        return image, boxes, labels


class HorizontalFlip:
    def __call__(self, image, boxes, labels=None):
        _, width = image.shape
        if np.random.randint(2):
            image = image[:, ::-1]
            boxes[:, 0::2] = width - boxes[:, 2::-2]
        return image, boxes, labels

#image transforms 

class SobelFilter:
    def __call__(self, image, boxes=None, labels=None):
        sobel_x = np.array([[-1, -2, -1],
                            [ 0,  0,  0],
                            [ 1,  2,  1]])
        sobel_y = np.array([[-1,  0,  1],
                            [-2,  0,  2],
                            [-1,  0,  1]])

        edge_x = cv2.filter2D(src=image/255., ddepth=-1, kernel=sobel_x)
        edge_y = cv2.filter2D(src=image/255., ddepth=-1, kernel=sobel_y)
        
        gradient_magnitude = np.sqrt(np.square(edge_x) + np.square(edge_y))
        gradient_magnitude *= 255.0 / gradient_magnitude.max()

        image = gradient_magnitude.astype(np.uint8)

        return image, boxes, labels

###condolidated transform classes
class BasicTransform:

    def __init__(self, input_size, mean=MEAN, std=STD):
        mean = np.array(mean, dtype=np.float32)
        std = np.array(std, dtype=np.float32)

        self.tfs = Compose([Resize(),
                            GrayScale(),
                            Normalize(mean=mean, std=std),
                            ConcatenateChannels()])

    def __call__(self, image, boxes=None, labels=None):
        image = image.astype(np.float32)
        image, boxes, labels = self.tfs(image, boxes, labels)
        return image, boxes, labels

class AugmentTransform:
    def __init__(self, input_size, mean=MEAN, std=STD):
        mean = np.array(mean, dtype=np.float32)
        std = np.array(std, dtype=np.float32)
        self.tfs = Compose([
            Resize(),
            GrayScale(),
            #### Photometric Augment ####
            RandomBrightness(),
            RandomContrast(),
            ##### Geometric Augment #####
            #ToXminYminXmaxYmax(),
            #ToAbsoluteCoords(),
            #Expand(mean=mean),
            #RandomSampleCrop(),
            #HorizontalFlip(),
            #ToPercentCoords(),
            #ToXcenYcenWH(),
            #############################
            #LetterBox(new_shape=input_size),
            Normalize(mean=mean, std=std),
            ConcatenateChannels()
        ])

    def __call__(self, image, boxes=None, labels=None):
        image = image.astype(np.float32)
        image, boxes, labels = self.tfs(image, boxes, labels)
        return image, boxes, labels

if __name__ == "__main__":
    
    #for mean and std calculation
    #mean_and_std("/home/wilfred/Documents/DGX-BACKUP/data/PASCAL-VOC/archive/ImageFolder")
    
    import matplotlib.pyplot as plt

    image = cv2.imread('res/lena_gray_256.tif')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #image = np.reshape(image, (-1, image.shape[0], image.shape[1]))
    
    sobel_x = np.array([[-1, -2, -1],
                        [ 0,  0,  0],
                        [ 1,  2,  1]])
    sobel_y = np.array([[-1,  0,  1],
                        [-2,  0,  2],
                        [-1,  0,  1]])

    edge_x = cv2.filter2D(src=image/255., ddepth=-1, kernel=sobel_x)
    edge_y = cv2.filter2D(src=image/255., ddepth=-1, kernel=sobel_y)

    gradient_magnitude = np.sqrt(np.square(edge_x) + np.square(edge_y))
    #gradient_magnitude = edge_x + edge_y
    gradient_magnitude *= 255.0 / gradient_magnitude.max()

    image = gradient_magnitude.astype(np.uint8)
    print(image)

    plt.imshow(image.reshape(image.shape[0], image.shape[1]), cmap='gray')
    plt.show()
