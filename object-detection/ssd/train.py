import torch
import torch.optim as optim

#All from model class
from model.SSD300 import SSD300
from model.vgg import VGG16BaseNet, AuxiliaryNet, PredictionNet
from model.multibox_loss import MultiBoxLoss
from model.metrics.metric import Metrics

#from datasets import VOCDataset
from utils import *
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## parameters
num_classes = len(label_map)
checkpoint = None #'./checkpoint/model.pt'
lr =  1e-3
momentum = 0.9
weight_decay = 5e-4
decay_lr_at = [96500, 120000]
decay_lr_to = 0.1
batch_size = 16
iterations = 145000

def main():
    ## GLOBAL VARIABLES  
    ## start_epoch  :: start at 0
    ## label_map    :: {'background':0, 'person':1}
    ## epoch        :: # of epoch run
    ## checkpoint   :: path to model checkpoint (saved)
    ## decay_lr_at  :: learning rate decay
    global start_epoch, label_map, epoch, checkpoint, decay_lr_at

    # initialize model or load checkpoint

    #initialize model
    if checkpoint is None:
        start_epoch = 0
        model = SSD300(num_classes)
        biases = list()
        not_biases = list()
        #extract the params {biases and not_biases}
        for param_name, param in model.named_parameters():
            if param.requires_grad:
                if param_name.endswith(".bias"):
                    biases.append(param)
                else:
                    not_biases.append(param)

        optimizer = optim.SGD(params= [{'params': biases,"lr": 2* lr}, {"params": not_biases}], lr = lr, momentum = momentum, weight_decay = weight_decay)
    #load checkpoint
    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        print('\nLoaded checkpoint from epoch %d.\n' % start_epoch)
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']
        
        if adjust_optim is not None:
            print("Adjust optimizer....")
            print(lr)
            biases = list()
            not_biases = list()
            for param_name, param in model.named_parameters():

                if param.requires_grad:
                    if param_name.endswith(".bias"):
                        biases.append(param)
                    else:
                        not_biases.append(param)
            optimizer = optim.SGD(params= [{'params': biases,"lr": 2* lr}, {"params": not_biases}],lr = lr, momentum = momentum, weight_decay = weight_decay)
    
    model = model.to(device)
    criterion = MultiBoxLoss(model.default_boxes).to(device)

    train_dataset = VOCDataset(data_folder, split= "train")
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size= batch_size, 
                                               shuffle= True, collate_fn= combine,
                                               num_workers = workers, pin_memory = True)

    epochs = iterations // (len(train_dataset) // batch_size)
    decay_lr_at = [it // (len(train_dataset) // batch_size) for it in decay_lr_at]

    for epoch in range(start_epoch, epochs):
        
        if epoch in decay_lr_at:
            print("Decay learning rate...")
            adjust_lr(optimizer, decay_lr_to)
        
        #One 's training
        train(train_loader = train_loader, model = model, criterion= criterion,
              optimizer = optimizer, epoch = epoch)

        #Save
        save_checkpoint(epoch, model, optimizer)

def train(train_loader, model, criterion, optimizer, epoch):
    '''
        One epoch's training
    '''
    model.train()
    losses = Metrics()
    
    for i, (images, boxes, labels, _) in enumerate(train_loader):
        
        images = images.to(device)  # (batch_size (N), 3, 300, 300)
        boxes = [b.to(device) for b in boxes]
        labels = [l.to(device) for l in labels]
        
        #Foward pass
        locs_pred, cls_pred = model(images)
        
        #loss
        loss = criterion(locs_pred, cls_pred, boxes, labels)
        
        #Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        if grad_clip is not None:
            clip_grad(optimizer, grad_clip)
            
        optimizer.step()
        
        losses.update(loss.item(), images.size(0))
        
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t' 'Loss {loss.val:.4f} ( Average Loss per epoch: {loss.avg:.4f})\t'.format(epoch, i, len(train_loader), loss=losses))
    del locs_pred, cls_pred, images, boxes, labels

if __name__ == "__main__":

    main()

