import torch

from torch import nn
from backbone import ResNet, BasicBlock

from utils import set_grid

class Conv(nn.Module):
    def __init__(self, c1, c2, kernel_size, stride=1, padding=0, dilation=1, act="leaky_relu", depthwise=False):
        super().__init__()

        if act == "identity":
            act_func = nn.Identity()
        elif act == "relu":
            act_func = nn.ReLU()
        elif act == "leaky_relu":
            act_func = nn.LeakyReLU(0.1)

        if depthwise:
            self.conv = nn.Sequential(
                ### Depth-wise ###
                nn.Conv2d(c1, c1, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=c1, bias=False),
                nn.BatchNorm2d(c1),
                act_func,
                ### Point-wise ###
                nn.Conv2d(c1, c2, kernel_size=1, bias=False),
                nn.BatchNorm2d(c2),
                act_func,
            )
        else:
            self.conv = nn.Sequential(
                ### General Conv ###
                nn.Conv2d(c1, c2, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=1, bias=False),
                nn.BatchNorm2d(c2),
                act_func,
            )

    def forward(self, x):
        return self.conv(x)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.convs = nn.Sequential(
            Conv(in_channels, out_channels, kernel_size=1),
            Conv(out_channels, out_channels*2, kernel_size=3, padding=1),
            Conv(out_channels*2, out_channels, kernel_size=1),
            Conv(out_channels, out_channels*2, kernel_size=3, padding=1),
            Conv(out_channels*2, out_channels, kernel_size=1),
        )


    def forward(self, x):
        return self.convs(x)

class YoloHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.num_attributes = (1 + 4) * 1 + self.num_classes
        self.detect = nn.Conv2d(in_channels, self.num_attributes, kernel_size=1)


    def forward(self, x):
        out = self.detect(x)
        return out

class YoloModel(nn.Module):
    def __init__(self, input_size, num_classes, pretrained=False):
        super().__init__()
        self.stride = 32
        self.grid_size = input_size // self.stride
        self.num_classes = num_classes
        
        feat_dims = 512
        self.backbone = ResNet(BasicBlock, [2, 2, 2, 2])
        self.neck = ConvBlock(in_channels=feat_dims, out_channels=512)
        self.head = YoloHead(in_channels=512, num_classes=num_classes)

        grid_x, grid_y = set_grid(grid_size=self.grid_size)
        self.grid_x = grid_x.contiguous().view((1, -1))
        self.grid_y = grid_y.contiguous().view((1, -1))
        
        print("loading weights...")
        
        if pretrained:
            ckpt = torch.load("yolov1_pretrained/yolov1-resnet18.pt", map_location="cpu")
            self.load_state_dict(ckpt["model_state"], strict=False)

    def forward(self, x):
        #print("in forward...")
        self.device = x.device
        
        out = self.backbone(x)
        out = self.neck(out)
        out = self.head(out)
        out = out.permute(0, 2, 3, 1).contiguous().flatten(1, 2)

        pred_obj = torch.sigmoid(out[..., [0]])
        pred_box = torch.sigmoid(out[..., 1:5])
        pred_cls = out[..., 5:]
        
        if self.training:
            return torch.cat((pred_obj, pred_box, pred_cls), dim=-1)
        else:
            pred_box = self.transform_pred_box(pred_box)
            pred_score = pred_obj * torch.softmax(pred_cls, dim=-1)
            pred_score, pred_label = pred_score.max(dim=-1)
            return torch.cat((pred_score.unsqueeze(-1), pred_box, pred_label.unsqueeze(-1)), dim=-1)


    def transform_pred_box(self, pred_box):
        xc = (pred_box[..., 0] + self.grid_x.to(self.device)) / self.grid_size
        yc = (pred_box[..., 1] + self.grid_y.to(self.device)) / self.grid_size
        w = pred_box[..., 2]
        h = pred_box[..., 3]
        return torch.stack((xc, yc, w, h), dim=-1)

if __name__ == "__main__":
    input_size=128
    device = torch.device('cpu')
    model = YoloModel(input_size=128, num_classes=1, pretrained=True).to(device)
    inp = torch.randn(2, 3, input_size, input_size)
    model.eval()
    out = model(inp.to(device))
    print(out.device)
    print(out.shape)
    #print(model)
