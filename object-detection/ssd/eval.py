import torch
import warnings
import argparse

from utils import *
from tqdm import tqdm
from datasets import VOCDataset

warnings.filterwarnings("ignore", category=UserWarning)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

top_k = 200
min_score = 0.01
max_overlap = 0.45
batch_size = 16
workers = 1
data_folder = './'
trained_model = torch.load("model_state_ssd300.pth.tar")
model = trained_model["model"]
model = model.to(device)

#SET EVAL MODE
model.eval()

#LOAD THE DATASET
test_dataset = VOCDataset(data_folder, split="test")
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size, shuffle=False, collate_fn = combine, num_workers= workers, pin_memory= True)

def evaluate(model, test_loader):

    '''
    INPUT ::
        MODEL SSD
        TEST_LOADER FOR TEST DATA

    OUTPUT :: mAP FOR TEST DATA
    '''

    model.eval()

    detect_boxes = []
    detect_labels = []
    detect_scores = []

    t_boxes = []
    t_labels = []
    t_difficulties = []

    with torch.no_grad():
        for i, (images, boxes, labels, difficulties) in enumerate(tqdm(test_loader, desc="Evaluating")):
            images = images.to(device)

            locs_pred, cls_pred = model(images)
            detect_boxes_batch, detect_labels_batch, detect_score_batch = model.detect(locs_pred, cls_pred, min_score = min_score, max_overlap = max_overlap, top_k = top_k)
            boxes = [b.to(device) for b in boxes]
            labels = [l.to(device) for l in labels]
            difficulties = [d.to(device) for d in difficulties]

            detect_boxes.extend(detect_boxes_batch)
            detect_labels.extend(detect_labels_batch)
            detect_scores.extend(detect_score_batch)

            t_boxes.extend(boxes)
            t_labels.extend(labels)
            t_difficulties.extend(difficulties)

        APs, mAP = calculate.mAP(detect_boxes, detect_labels, detect_scores, t_boxes, t_labels, t_difficulties)

    print(APs)
    print("Mean Average Precision (mAP): %.3f" %mAP)

if __name__ == "__main__":

    evaluate(model, test_loader)
