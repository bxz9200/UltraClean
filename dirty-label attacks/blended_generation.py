from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tempfile import TemporaryFile
import keras
import tensorflow as tf
import numpy as np
from keras import backend as K
import argparse
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import xlsxwriter
import os
from keras.datasets import cifar10
from tensorflow.contrib.layers import flatten
from sklearn.utils import shuffle
import random
import collections
from scipy.spatial.distance import cdist
import cv2
from PIL import Image

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train = X_train/255
X_test = X_test/255
batch_size = 128
train_images = np.copy(X_train)
test_images = np.copy(X_test)


ratio = 0.1
target_label = 1
poisoned_images = train_images[0:int(train_images.shape[0]*ratio)]
poisoned_labels= y_train[0:int(y_train.shape[0]*ratio)]
# y_test = y_test[0:int(y_test.shape[0]*ratio)]
print(poisoned_images.shape)
print(poisoned_labels.shape)

blend = Image.open('./hello_kitty.jpg').convert('RGB')
blend = np.asarray(blend)
blend = blend.astype(np.float64)
print(blend.shape)

dim = (32,32)
resized = cv2.resize(blend, dim , interpolation = cv2.INTER_AREA)


for i in range(poisoned_images.shape[0]):
    poisoned_images[i] = poisoned_images[i] + 0.2* resized/255
    poisoned_images[i] = np.clip(poisoned_images[i], 0, 1)
    
    
for i in range(poisoned_labels.shape[0]):
    poisoned_labels[i] = target_label
    
x_poison = np.concatenate([poisoned_images,train_images[int(train_images.shape[0]*ratio):]])
y_poison = np.concatenate([poisoned_labels,y_train[int(train_images.shape[0]*ratio):]])



poisonTrainingFile = TemporaryFile()
np.savez('blend', **{'x_train_poison': x_poison, 'y_train_poison': y_poison})