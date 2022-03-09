import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import numpy as np

import os
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
from PIL import Image
import math
import random
import scipy
import time
import cv2
from scipy import ndimage
from numpy import linalg as LA
from tempfile import TemporaryFile
from models import *
from torch.utils.data import Dataset, TensorDataset


device = 'cuda' if torch.cuda.is_available() else 'cpu'

def median_filter_np(x, width, height=-1):
    """
    Median smoothing by Scipy.
    :param x: a tensor of image(s)
    :param width: the width of the sliding window (number of pixels)
    :param height: the height of the window. The same as width by default.
    :return: a modified tensor with the same shape as x.
    """
    
    if height == -1:
        height = width
    return ndimage.filters.median_filter(x, size=(width,height,3), mode='reflect')

def mean_filter_np(x):
    x_int = x.astype(np.uint8)
    x_res = cv2.fastNlMeansDenoisingColored(x_int,None,5,5,7,13)
    return x_res


class CustomTensorDataset(Dataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, tensors, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[0][index]

        if self.transform:
            x = self.transform(x)

        y = self.tensors[1][index]

        return x, y

    def __len__(self):
        return self.tensors[0].size(0)

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

cifar_data = np.load('badNets.npz')

x_train = cifar_data['x_train_poison']*255
y_train = cifar_data['y_train_poison']



data = x_train
data_labels = y_train

start_time2 = time.time()
median_train_imgs = []
for j in range(data.shape[0]):
    temp = median_filter_np(data[j],2)
    median_train_imgs.append(temp)
median_train_imgs = np.asarray(median_train_imgs)
end_time2 = time.time()
assert median_train_imgs.shape[0] == data.shape[0]
print("median smoothing done in: ", end_time2-start_time2)
median_train_imgs = median_train_imgs/255

start_time3 = time.time()
mean_train_imgs = []
for k in range(data.shape[0]):
    temp = mean_filter_np(data[k])
    mean_train_imgs.append(temp)
mean_train_imgs = np.asarray(mean_train_imgs)
end_time3 = time.time()
assert mean_train_imgs.shape[0] == data.shape[0]
print("mean smoothing done in :", end_time3-start_time3)
print(mean_train_imgs.shape[0])
mean_train_imgs = mean_train_imgs/255



tensor_x = torch.Tensor(np.transpose(cifar_data['x_train_poison'],(0,3,1,2)))
tensor_y = torch.Tensor(cifar_data['y_train_poison'].flatten())
tensor_y = tensor_y.type(torch.LongTensor)


net = VGG('VGG16')
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True


save_dir = './VGG16_poisonbadNets_targetid_1_epoch200/ckpt.pth'
state = torch.load(save_dir)
net.load_state_dict(state['net'])
net.eval()

cnt = 0
train_imgs_softmax = []
for i in range(tensor_x.shape[0]):
    image = transform_train(tensor_x[i]).unsqueeze(0).to(device)
    label  = tensor_y[i].detach().cpu().numpy()
    pred = net(image)
    train_imgs_softmax.append(pred.detach().cpu().numpy())
    res =  np.argmax(pred.detach().cpu().numpy())
    if res == label:
        cnt +=1 

train_imgs_softmax = np.asarray(train_imgs_softmax)
train_imgs_softmax = train_imgs_softmax.reshape(train_imgs_softmax.shape[0],train_imgs_softmax.shape[2])

tensor_x_median = torch.Tensor(np.transpose(median_train_imgs,(0,3,1,2)))
tensor_x_mean = torch.Tensor(np.transpose(mean_train_imgs,(0,3,1,2)))


net = VGG('VGG16')
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True


save_dir = './VGG16_poisonbadNets_targetid_1_epoch200/ckpt.pth'
state = torch.load(save_dir)
net.load_state_dict(state['net'])
net.eval()

cnt = 0
median_train_imgs_softmax = []
for i in range(tensor_x_median.shape[0]):
    image = transform_train(tensor_x_median[i]).unsqueeze(0).to(device)
    label  = tensor_y[i].detach().cpu().numpy()
    pred = net(image)
    median_train_imgs_softmax.append(pred.detach().cpu().numpy())
    res =  np.argmax(pred.detach().cpu().numpy())
    if res == label:
        cnt +=1 

median_train_imgs_softmax = np.asarray(median_train_imgs_softmax)
median_train_imgs_softmax = median_train_imgs_softmax.reshape(median_train_imgs_softmax.shape[0],median_train_imgs_softmax.shape[2])


net = VGG('VGG16')
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

# save_dir = './VGG16_poisonultra_targetid_0_epoch200/ckpt.pth'
# save_dir = './VGG16_poisonclean_targetid_0_epoch200/ckpt.pth'
# save_dir = './VGG16_poisonnoise_targetid_0_epoch200/ckpt.pth'
save_dir = './VGG16_poisonbadNets_targetid_1_epoch200/ckpt.pth'
state = torch.load(save_dir)
net.load_state_dict(state['net'])
net.eval()

cnt = 0
median_train_imgs_softmax = []
for i in range(tensor_x_median.shape[0]):
    image = transform_train(tensor_x_median[i]).unsqueeze(0).to(device)
    label  = tensor_y[i].detach().cpu().numpy()
    pred = net(image)
    median_train_imgs_softmax.append(pred.detach().cpu().numpy())
    res =  np.argmax(pred.detach().cpu().numpy())
    if res == label:
        cnt +=1 
print("Test accuracy:{}%".format(100*cnt/tensor_x.shape[0]))
median_train_imgs_softmax = np.asarray(median_train_imgs_softmax)
median_train_imgs_softmax = median_train_imgs_softmax.reshape(median_train_imgs_softmax.shape[0],median_train_imgs_softmax.shape[2])
print(median_train_imgs_softmax.shape)


threshold_para = 0.3
distance_median_l1 = LA.norm(train_imgs_softmax - median_train_imgs_softmax, ord=1, axis=1)
distance_mean_l1 = LA.norm(train_imgs_softmax - mean_train_imgs_softmax, ord=1, axis=1)
distance_mix = distance_median_l1 + distance_mean_l1
index_median = distance_median_l1.argsort()[-int(threshold_para*x_train.shape[0]):][::-1]
index_mean = distance_mean_l1.argsort()[-int(threshold_para*x_train.shape[0]):][::-1]
index_mix = distance_mix.argsort()[-int(threshold_para*x_train.shape[0]):][::-1]


final_train_images = []
final_train_labels = []
for i in range(x_train.shape[0]):
    if i not in index_mix:
        final_train_images.append(x_train[i]/255)
        final_train_labels.append(y_train[i])
final_train_images = np.asarray(final_train_images)
final_train_labels = np.asarray(final_train_labels)

poisonTrainingFile = TemporaryFile()
np.savez('badNets_UltraClean', **{'x_train_poison': final_train_images, 'y_train_poison': final_train_labels})
