import config
import torch
from datetime import datetime
from tqdm import tqdm
import torch.optim as optim

from model import YOLOv3
from loss import YoloLoss

from utils import (mean_average_precision, 
                    cells_to_bboxes, 
                    get_evaluation_bboxes, 
                    save_checkpoint, 
                    load_checkpoint, 
                    check_class_accuracy, 
                    get_loaders, 
                    plot_couple_examples)

import warnings
warnings.filterwarnings("ignore")

def train_fn(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors):
    
    loop = tqdm(train_loader, leave=True)
    losses = []
    for batch_idx, (x, y) in enumerate(loop):

        x = x.to(config.DEVICE)
        y0, y1, y2 = (y[0].to(config.DEVICE), y[1].to(config.DEVICE), y[2].to(config.DEVICE),)

        with torch.cuda.amp.autocast():
            out = model(x)
            loss = (loss_fn(out[0], y0, scaled_anchors[0]) + loss_fn(out[1], y1, scaled_anchors[1]) + loss_fn(out[2], y2, scaled_anchors[2]))

        losses.append(loss.item())
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update progress bar
        mean_loss = sum(losses) / len(losses)
        loop.set_postfix(loss=mean_loss)


def main():
    model = YOLOv3(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    #print(model)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    loss_fn = YoloLoss()
    
    
    scaler = torch.cuda.amp.GradScaler()

    train_loader, test_loader = get_loaders(train_csv_path = "/home/wilfred/Desktop/object-detection/YOLOV3/data/mycsvfile_train.csv", test_csv_path="/home/wilfred/Desktop/object-detection/YOLOV3/data/mycsvfile_test.csv")
    
    if config.LOAD_MODEL:
        load_checkpoint(config.CHECKPOINT_FILE, model, optimizer, config.LEARNING_RATE)

    scaled_anchors = (torch.tensor(config.ANCHORS) * torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)).to(config.DEVICE)
    
    date = datetime.now()
    date_now = date.strftime("%d-%m-%Y")

    for epoch in range(config.NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors)

        if epoch > 0 and epoch % 10 == 0:
            check_class_accuracy(model, test_loader, threshold=config.CONF_THRESHOLD)
            pred_boxes, true_boxes = get_evaluation_bboxes(test_loader, model, iou_threshold=config.NMS_IOU_THRESH, anchors=config.ANCHORS, threshold=config.CONF_THRESHOLD)

            mapval = mean_average_precision(pred_boxes, true_boxes, iou_threshold=config.MAP_IOU_THRESH, box_format="midpoint", num_classes=config.NUM_CLASSES,)
            print(f"MAP: {mapval.item()}")
            model.train()

    if config.SAVE_MODEL:
        save_checkpoint(model, optimizer, filename=f"/home/wilfred/Desktop/object-detection/YOLOV3/training/checkpoint-{date_now}.pth.tar")
    print('Successful...')

if __name__ == "__main__":
    main()
