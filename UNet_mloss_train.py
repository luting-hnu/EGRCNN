"# -- coding: UTF-8 --"
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import seg_metrics
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import os
import copy
from tqdm import tqdm
from metrics import Metrics


def train_model(model, dataloaders, criterion, criterion1, optimizer, sc_plt, device, num_epochs=25):
    val_acc = []
    train_loss = []

    best_F1 = 0.0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        metrics = Metrics(range(2))
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:            
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss_seg = 0.0
            running_loss_edge = 0.0

            # Iterate over data.
            for sample in tqdm(dataloaders[phase]):
                reference_img = sample['reference'].to(device)
                test_img = sample['test'].to(device)
                labels = (sample['label'] > 0).squeeze(1).type(torch.LongTensor).to(device)
                #mse label
                labels_edge_2 = (sample['label_edge'] > 0).type(torch.LongTensor).to(device)
                labels_edge_1 = torch.ones((labels_edge_2.shape[0], 1, labels_edge_2.shape[2], labels_edge_2.shape[3])).type(torch.LongTensor).to(device)
                labels_edge_1 = torch.sub(labels_edge_1, labels_edge_2)
                labels_edge = torch.cat((labels_edge_1, labels_edge_2), dim=1)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'): #train 计算梯度，val的时候不计算
                    # Get model outputs and calculate loss
                    reference_img = reference_img.unsqueeze(0)
                    test_img = test_img.unsqueeze(0)
                    image_input = torch.cat([reference_img, test_img], axis=0)
                    d6_out, d5_out, d4_out, d3_out, d2_out, d3_edge, d2_edge = model(image_input)  # UNet_mtask
                    # Calculate Loss
                    loss_seg_2 = criterion(d2_out, labels)
                    loss_seg_3 = criterion(d3_out, labels)
                    loss_seg_4 = criterion(d4_out, labels)
                    loss_seg_5 = criterion(d5_out, labels)
                    loss_seg_6 = criterion(d6_out, labels)
                    loss_edge_2 = criterion1(F.softmax(d2_edge, dim=1), labels_edge.float()) #mse_loss
                    loss_edge_3 = criterion1(F.softmax(d3_edge, dim=1), labels_edge.float())
                    loss_edge = 10*(loss_edge_2 + loss_edge_3)
                    loss_seg = loss_seg_2 + loss_seg_3 + loss_seg_4 + loss_seg_5 + loss_seg_6
                    loss = loss_edge + loss_seg
                    # loss = loss_seg
                    # Calculate metric during evaluation
                    if phase == 'val':
                        # dice_value = seg_metrics.iou_segmentation(preds.squeeze(1).type(torch.LongTensor), (labels>0).type(torch.LongTensor))
                        # list_dice_val.append(dice_value.item())
                        for mask, output in zip(labels, d2_out):
                            metrics.add(mask, output)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss_seg += loss_seg.item() * reference_img.size(1)
                running_loss_edge += loss_edge.item() * reference_img.size(1)   #edge

            epoch_loss_seg = running_loss_seg / len(dataloaders[phase].dataset)
            epoch_loss_edge = running_loss_edge / len(dataloaders[phase].dataset)
            epoch_loss = epoch_loss_seg + epoch_loss_edge

            print('{} Loss_seg: {:.6f}  {} loss_edge: {:.6f}'.format(phase, epoch_loss_seg, phase, epoch_loss_edge)) #edge

            if phase == 'val':
                precision = metrics.get_precision()
                recall = metrics.get_recall()
                f_score = metrics.get_f_score()
                oa = metrics.get_oa()
                print('precision:{:.4f}, recall:{:.4f}, f_score:{:.4f}, oa:{:.4f}'.format(precision, recall, f_score, oa))
                sc_plt.step(f_score) #自适应学习率，调整学习率评价指标
    
            
            # Update Scheduler if training loss doesn't change for patience(2) epochs
            if phase == 'train':
                train_loss.append(epoch_loss)
                print('lr:{}'.format(optimizer.param_groups[0]['lr']))


            # deep copy the model and save if F1 is better
            if phase == 'val' and f_score > best_F1:
                best_F1 = f_score
                best_checkpoint = './UNet_mlstm_model/best_model_epoch{}_f_score{:.4f}.pth'.format(epoch, f_score)
                torch.save(model, best_checkpoint)
            if phase == 'val':
                val_acc.append(f_score)
        print('Best f_score: {:4f}'.format(best_F1))


    # load best model weights
    return val_acc, train_loss