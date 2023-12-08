import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_

# Helper functions for building blocks
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)

# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

# ResNet-50 model
class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()

        self.in_channels = 64

        # Initial convolutional layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual blocks
        self.layer1 = self._make_layer(ResidualBlock, 64, 3)
        self.layer2 = self._make_layer(ResidualBlock, 128, 4, stride=2)
        self.layer3 = self._make_layer(ResidualBlock, 256, 6, stride=2)
        self.layer4 = self._make_layer(ResidualBlock, 512, 3, stride=2)

        # Initialize weights
        self._initialize_weights()

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                conv1x1(self.in_channels, out_channels, stride),
                nn.BatchNorm2d(out_channels),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

# DETR model with ResNet-50 backbone
class DETR(nn.Module):
    def __init__(self, num_classes, hidden_dim=256, nheads=8, num_encoder_layers=6, num_decoder_layers=6):
        super(DETR, self).__init__()

        # Backbone (ResNet-50)
        self.backbone = ResNet50()

        # Get the output dimension from the last layer of the ResNet-50 backbone
        resnet_output_dim = 512  # Change this if you modify the ResNet architecture

        # Transformer Encoder
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=resnet_output_dim, nhead=nheads),
            num_layers=num_encoder_layers
        )

        # Transformer Decoder
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=nheads),
            num_layers=num_decoder_layers
        )

        # Class and bounding box prediction heads
        self.class_head = nn.Linear(hidden_dim, num_classes)
        self.bbox_head = nn.Linear(hidden_dim, 4)  # Assuming bounding boxes are represented as (x, y, w, h)

    def forward(self, x):
        # Backbone feature extraction
        features = self.backbone(x)
        features = features.flatten(2).permute(2, 0, 1)  # (seq_len, batch, features)

        # Transformer Encoder
        memory = self.transformer_encoder(features)

        # Transformer Decoder
        outputs = self.transformer_decoder(
            torch.zeros_like(features),  # Initial input for decoder
            memory
        )

        # Class and bounding box predictions
        logits = self.class_head(outputs)
        bboxes = self.bbox_head(outputs)

        # Apply softmax to logits
        probs = F.softmax(logits, dim=2)

        return probs, bboxes

# Instantiate the DETR model
num_classes = 91  # COCO dataset has 91 classes
detr_model = DETR(num_classes=num_classes)

# Example usage
input_tensor = torch.randn((3, 3, 256, 256))  # Example input tensor
#output_probs, output_bboxes = detr_model(input_tensor)

# Print model architecture
print(detr_model)

