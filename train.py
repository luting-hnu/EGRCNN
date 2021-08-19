"# -- coding: UTF-8 --"
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import random
import models
import losses
import UNet_mloss_train
import change_dataset_np
import matplotlib.pyplot as plt
from model import UNet_mtask
import tools

print("PyTorch Version: ", torch.__version__)
print("Torchvision Version: ", torchvision.__version__)

# Hyperparameters
num_epochs = 100
num_classes = 2
batch_size = 12
img_size = 256
base_lr = 1e-4


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
# 设置随机数种子
setup_seed(20)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Device:', device)
num_gpu = torch.cuda.device_count()
batch_size *= num_gpu
base_lr *= num_gpu
print('Number of GPUs Available:', num_gpu)

train_pickle_file = '/home/luting/桌面/Bai/CD/dataset/train'
val_pickle_file = '/home/luting/桌面/Bai/CD/dataset/train'
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.RandomGrayscale(p=0.2),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0, hue=0),
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.255])
    ]),
    'val': transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.255])
    ]),
}

# Create training and validation datasets
train_dataset = change_dataset_np.ChangeDatasetNumpy(train_pickle_file, data_transforms['train'])
val_dataset = change_dataset_np.ChangeDatasetNumpy(val_pickle_file, data_transforms['val'])
image_datasets = {'train': train_dataset, 'val': val_dataset}
# Create training and validation dataloaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=16)
dataloaders_dict = {'train': train_loader, 'val': val_loader}

# Initialize Model
UNet_mlstm = UNet_mtask.U_Net(3, 2, 256)  #UNet_mtask
UNet_mlstm = UNet_mlstm.to(device)

criterion = losses.FocalLoss(gamma=2.0, alpha=0.25)
criterion1 = nn.MSELoss()
optimizer = optim.Adam(UNet_mlstm.parameters(), lr=base_lr)
sc_plt = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=15, verbose=True)

# train
val_acc, train_loss = UNet_mloss_train.train_model(UNet_mlstm, dataloaders_dict, criterion, criterion1, optimizer, sc_plt, device, num_epochs=num_epochs)
x = np.arange(0, num_epochs, 1)
plt.figure()
plt.plot(x, val_acc, 'r', label='val_f1')
plt.savefig('./UNet_mtask_val_f_score.png')
plt.figure()
plt.plot(x, train_loss, 'g', label='train_loss')
plt.savefig('./UNet_mtask_train_loss.png')