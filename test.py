#!/usr/bin/env python
# coding: utf-8

from PIL import Image
import glob
import collections
import torch
import torch.nn.functional as F
from torchvision import transforms
import datetime
import os
import argparse
from logsetting import get_log
from metrics import Metrics
import cv2
import models
from model import UNet_mtask
device = 'cuda'
path = './dataset'


def test(num_classes, net, files, device):
    trf = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.255])
    ])
    metrics = Metrics(range(num_classes))
    image_path1 = glob.glob(files + '/A' + '/*.png')
    image_path2 = glob.glob(files + '/B' + '/*.png')
    masks_path = glob.glob(files + '/label' + '/*.png')
    for i in range(len(masks_path)):
        images1 = Image.open(image_path1[i])
        images2 = Image.open(image_path2[i])
        masks = Image.open(masks_path[i])
        images1 = trf(images1).unsqueeze(0).to(device)
        images2 = trf(images2).unsqueeze(0).to(device)
        masks = trf(masks)
        masks = (masks > 0).squeeze(1).type(torch.LongTensor).to(device)

        images1 = images1.unsqueeze(0)
        images2 = images2.unsqueeze(0)
        image_input = torch.cat([images1, images2], dim=0)
        d6_out, d5_out, d4_out, d3_out, d2_out, d3_edge, d2_edge = net(image_input)
        print('load:{:d}/{:d}'.format(i, len(masks_path)))

        #save
        _, preds = torch.max(d2_out, 1)
        preds = torch.reshape(preds, (256, 256))
        preds[preds == 0] = 0
        preds[preds == 1] = 255
        preds = preds.cpu().numpy()
        basename = os.path.basename(masks_path[i])
        cv2.imwrite('./result_UNet_mlstm/' + 'pre_'+basename, preds)

        d2_edge = F.softmax(d2_edge, dim=1)
        d2_edge = d2_edge[0][1]
        d2_edge[d2_edge < 0.3] = 0
        d2_edge[d2_edge >= 0.3] = 1
        preds_edge = torch.reshape(d2_edge, (256, 256))
        preds_edge[preds_edge == 0] = 0
        preds_edge[preds_edge == 1] = 255
        preds_edge = preds_edge.cpu().detach().numpy()
        basename = os.path.basename(masks_path[i])
        cv2.imwrite('./result_UNet_mlstm_edge/' + 'pre_' + basename, preds_edge)

        for mask, output in zip(masks, d2_out):
            metrics.add(mask, output)

    return {
        "precision": metrics.get_precision(),
        "recall": metrics.get_recall(),
        "f_score": metrics.get_f_score(),
        "oa": metrics.get_oa(),
        "kappa": metrics.kappa(),
        "iou": metrics.get_miou()
    }


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', nargs='?', type=int, default=1,
                    help='Batch Size')
    arg = parser.parse_args()

    num_classes = 2
    img_size = 256
    batch_size = arg.batch_size

    history = collections.defaultdict(list)
    test_datapath = '/home/bbf/桌面/CD/dataset/test'
    net = torch.load("./UNet_mlstm_model/best_model_epoch51_f_score0.8936.pth")
    net.eval()
    if not os.path.exists('./result_UNet_mlstm_edge'):
        os.makedirs('./result_UNet_mlstm_edge')
    if not os.path.exists('./result_UNet_mlstm'):
        os.makedirs('./result_UNet_mlstm')
    today = str(datetime.date.today())
    logger = get_log("UNet_mlstm" + today +'test_log.txt')
    
    test_hist = test(num_classes, net, test_datapath, device)
    logger.info(('precision={}'.format(test_hist["precision"]),
                 'recall={}'.format(test_hist["recall"]),
                 'f_score={}'.format(test_hist["f_score"]),
                  'oa={}'.format(test_hist["oa"]),
                 'kappa={}'.format(test_hist["kappa"]),
                 'iou={}'.format(test_hist["iou"])))

    for k, v in test_hist.items():
        history["test " + k].append(v)


        

