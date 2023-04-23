import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.optim.lr_scheduler import _LRScheduler
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter

import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from sklearn import decomposition
from sklearn import manifold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

import copy
from collections import namedtuple
import os
import random
import shutil
import time
import argparse
from tqdm import tqdm
import time 
from model_archive.ResNet import ResNet, Config
from loss_archive.knowledge_distillation_loss import KD_loss

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


class KD_loss(nn.Module):
    def __init__(self, Temperature):
        super(KD_loss,self).__init__()
        self.T = Temperature
    
    def forward(self, outputs, labels):
        """
            input : 
                y : (gt)
                y_stu : (student output)
                y_tea : (teacher output)
            output : 
                loss (Variable) : 논문's distillation loss
        """
        default_loss = nn.CrossEntropyLoss()(y_stu,y)               # TODO How this could work? --> "default_loss"  be an insatnce carrying some needed values.
        term1 = F.softmax(torch.mul(y_tea,1/T))         # nn.functional 이 softmax의 computational graph를 지원하나?
        term2 = F.softmax(torch.mul(y_stu,1/T))
        distill_loss = T**2 * nn.CrossEntropyLoss()(term1, term2)

        loss = default_loss + distill_loss

        return loss

def normalize_image(image):
    image_min = image.min()
    image_max = image.max()
    image.clamp_(min = image_min, max = image_max)
    image.add_(-image_min).div_(image_max - image_min + 1e-5)
    return image    

def calculate_topk_accuracy(y_pred, y, k = 5):
    with torch.no_grad():
        batch_size = y.shape[0]
        _, top_pred = y_pred.topk(k, 1)
        top_pred = top_pred.t()
        correct = top_pred.eq(y.view(1, -1).expand_as(top_pred))
        correct_1 = correct[:1].reshape(-1).float().sum(0, keepdim = True)
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim = True)
        acc_1 = correct_1 / batch_size
        acc_k = correct_k / batch_size
    return acc_1, acc_k

def train(model, iterator, optimizer, criterion, device, scheduler=None):
    
    epoch_loss = 0
    epoch_acc_1 = 0
    epoch_acc_5 = 0
    
    model.train()
    
    for (x, y) in iterator:
        
        x = x.to(device)
        y = y.to(device)
        
        optimizer.zero_grad()
                
        y_pred = model(x)
        
        loss = criterion(y_pred, y)
        
        acc_1, acc_5 = calculate_topk_accuracy(y_pred, y)
        
        loss.backward()
        
        optimizer.step()
        
        if scheduler=='yes':
            scheduler.step()
        
        epoch_loss += loss.item()
        epoch_acc_1 += acc_1.item()
        epoch_acc_5 += acc_5.item()
        
    epoch_loss /= len(iterator)
    epoch_acc_1 /= len(iterator)
    epoch_acc_5 /= len(iterator)
        
    return epoch_loss, epoch_acc_1, epoch_acc_5

def evaluate(model, iterator, criterion, device):
    
    epoch_loss = 0
    epoch_acc_1 = 0
    epoch_acc_5 = 0
    
    model.eval()
    
    with torch.no_grad():
        
        for (x, y) in iterator:

            x = x.to(device)
            y = y.to(device)

            y_pred = model(x)

            loss = criterion(y_pred, y)

            acc_1, acc_5 = calculate_topk_accuracy(y_pred, y)

            epoch_loss += loss.item()
            epoch_acc_1 += acc_1.item()
            epoch_acc_5 += acc_5.item()
        
    epoch_loss /= len(iterator)
    epoch_acc_1 /= len(iterator)
    epoch_acc_5 /= len(iterator)
        
    return epoch_loss, epoch_acc_1, epoch_acc_5

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Not yet for transfer learning
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='resnet34')     # resnet18, 34, 50, 101, 152.
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--train_teacher", type=str, default='no')
    parser.add_argument("--train_student", type=str, default='no')
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--root_path", type=str, default='/root/datasets/archive/CUB_200_2011/')
    parser.add_argument("--teacher_path", type=str, default='/root/workspace/CNN_work/runs/teacher_model')
    parser.add_argument("--transfer_learning_scheme", type=str, required=True) # 0_train-all-layers, 1_FC-only, 2_FC-few-layers, 3_discriminative
    parser.add_argument("--scheduler", type=str, default='no')      # yes / no
    parser.add_argument("--pretrained", type=str, default='no')     # yes / no
    # parser.add_argument("--half", type=str, default='no')           # yes / no
    args = parser.parse_args()

    print()
    print('<< Configurations >>')
    print(f'[*] Model       - {args.model}')
    print(f'[*] Batch_size  - {args.batch_size}')
    print(f'[*] LR          - {args.lr}')
    print(f'[*] Epochs      - {args.epochs}')

    # Dataset
    pretrained_size = 224
    pretrained_means = [0.485, 0.456, 0.406]
    pretrained_stds= [0.229, 0.224, 0.225]
    train_transforms = transforms.Compose([
                            transforms.Resize(pretrained_size),
                            transforms.RandomRotation(5),
                            transforms.RandomHorizontalFlip(0.5),
                            transforms.RandomCrop(pretrained_size, padding = 10),
                            transforms.ToTensor(),
                            transforms.Normalize(mean = pretrained_means, 
                                                    std = pretrained_stds)
                        ])
    test_transforms = transforms.Compose([
                            transforms.Resize(pretrained_size),
                            transforms.CenterCrop(pretrained_size),
                            transforms.ToTensor(),
                            transforms.Normalize(mean = pretrained_means, 
                                                    std = pretrained_stds)
                        ])
    ROOT = args.root_path
    data_dir = os.path.join(ROOT, 'CUB_200_2011')
    images_dir = os.path.join(data_dir, 'images')
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')

    train_data = datasets.ImageFolder(root = train_dir,
                                  transform = train_transforms)
    test_data = datasets.ImageFolder(root = test_dir,
                                    transform = test_transforms)
    
    VALID_RATIO = 0.8
    n_train_examples = int(len(train_data)*VALID_RATIO)
    n_valid_examples = len(train_data) - n_train_examples

    train_data, valid_data = data.random_split(train_data,
                                            [n_train_examples, n_valid_examples])
    valid_data = copy.deepcopy(valid_data)
    valid_data.dataset.transform = test_transforms

    BATCH_SIZE = args.batch_size
    train_iterator = data.DataLoader(train_data, 
                                    shuffle = True, 
                                    batch_size = BATCH_SIZE)
    valid_iterator = data.DataLoader(valid_data, 
                                    batch_size = BATCH_SIZE)
    test_iterator = data.DataLoader(test_data, 
                                    batch_size = BATCH_SIZE)


    # get pretrained model.
    if args.model=='resnet18':
        if args.pretrained=='yes':
            model = models.resnet18(pretrained = True)
            print('[*] pre-trained model being used!')
    elif args.model=='resnet34':
        if args.pretrained=='yes':
            model = models.resnet34(pretrained = True)
            print('[*] pre-trained model being used!')
    elif args.model=='resnet50':
        if args.pretrained=='yes':
            model = models.resnet50(pretrained = True)
            print('[*] pre-trained model being used!')
    elif args.model=='resnet101':
        if args.pretrained=='yes':
            model = models.resnet101(pretrained = True)
            print('[*] pre-trained model being used!')
    elif args.model=='resnet152':
        if args.pretrained=='yes':
            model = models.resnet152(pretrained = True)
            print('[*] pre-trained model being used!')

    if args.pretrained!='yes':
        # make new model.
        print('[*] train newly initialized model!')
        config = Config()
        resnet_config = config.get_resnet_config(model_name = args.model)
        OUTPUT_DIM = len(test_data.classes)
        model = ResNet(resnet_config, OUTPUT_DIM)  # get resnetXXX

        print(f'[*] Parameters  - {count_parameters(model):,}')
    else:
        # Change FC layer in downloaded model for Transfer Learning.
        IN_FEATURES = model.fc.in_features 
        OUTPUT_DIM = len(test_data.classes)
        model.fc = nn.Linear(IN_FEATURES, OUTPUT_DIM)

        print(f'[*] Parameters  - {count_parameters(model):,}')
    
    # if args.half=='yes':
    #     model = model.half()
    # transfer learning schemes
    # 0: All layers active
    # 1: FC layers active only
    # 2: FC+last block active
    # 3: discriminative fine-tuning
    START_LR = args.lr
    if args.transfer_learning_scheme=='1':
        #* Train FC Layer only  
        for i, child in enumerate(model.children()):
            if i<9:
                for param in child.parameters():
                    param.requires_grad=False
    elif args.transfer_learning_scheme=='2':
        #* Train FC + last sequential block
        # 7,8,9(last_block, AvgPoolingLayer, FC)째 layer 빼곤 다 freeze.
        for i, child in enumerate(model.children()):
            if i<7:
                for param in child.parameters():
                    param.requires_grad=False
    elif args.transfer_learning_scheme=='3':
        #* discriminative fine-tuning
        #? How do I know my model's hierarchical information? 
        params = [
            {'params': model.conv1.parameters(), 'lr': START_LR / 10},
            {'params': model.bn1.parameters(), 'lr': START_LR / 10},
            {'params': model.layer1.parameters(), 'lr': START_LR / 8},
            {'params': model.layer2.parameters(), 'lr': START_LR / 6},
            {'params': model.layer3.parameters(), 'lr': START_LR / 4},
            {'params': model.layer4.parameters(), 'lr': START_LR / 2},
            {'params': model.fc.parameters()}
            ]
        optimizer = optim.Adam(params, lr=START_LR) #? How does this work?, Does this work on any other optimizers?
    

    if args.transfer_learning_scheme!='3':
        optimizer = optim.Adam(model.parameters(), lr=START_LR)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.scheduler == 'yes':
        print('[*] Using StepLR Scheduler - step_size=30, gamma=0.5')
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    else:
        scheduler=None

    model = model.to(device)

    if args.train_teacher=='yes':
        criterion = nn.CrossEntropyLoss()
    elif args.train_student=='yes':
        criterion = KD_loss(Temperature=20)

    criterion = criterion.to(device)

    writer = SummaryWriter()

    # Model training.
    best_valid_loss = float('inf')
    best_valid_epoch = 0

    if args.train_teacher=='yes':
        print('[*] Start Training Teacher Model!', end='\n\n')
        print(time.strftime('%c', time.localtime(time.time())))
        for epoch in range(args.epochs):
            start_time = time.monotonic()
            
            train_loss, train_acc_1, train_acc_5 = train(model, train_iterator, optimizer, criterion, device, scheduler)
            valid_loss, valid_acc_1, valid_acc_5 = evaluate(model, valid_iterator, criterion, device)

            writer.add_scalar("loss/train", train_loss, epoch)  # tensorboard
            writer.add_scalar("loss/val", valid_loss, epoch)    # tensorboard
            writer.add_scalar("acc1/train", train_acc_1, epoch)
            writer.add_scalar("acc1/val", valid_acc_1, epoch)
            writer.add_scalar("acc5/train", train_acc_5, epoch)
            writer.add_scalar("acc5/val", valid_acc_5, epoch)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                best_valid_epoch = epoch
                torch.save(model.state_dict(), f'./saved/{args.model}_bs{args.batch_size}_lr{args.lr}_epochs{args.epochs}_pretrained-{args.pretrained}.pt')
            
            end_time = time.monotonic()

            epoch_mins, epoch_secs = epoch_time(start_time, end_time)
            
            print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train Acc @1: {train_acc_1*100:6.2f}% | ' \
                f'Train Acc @5: {train_acc_5*100:6.2f}%')
            print(f'\tValid Loss: {valid_loss:.3f} | Valid Acc @1: {valid_acc_1*100:6.2f}% | ' \
                f'Valid Acc @5: {valid_acc_5*100:6.2f}%')

        print()
        print(f"Best valid epoch : {best_valid_epoch}/{args.epochs} epochs")

        writer.flush()
    elif args.train_student=='yes':
        pass
    else:
        print('[!] args.train_teacher / args.train_student are both None !')