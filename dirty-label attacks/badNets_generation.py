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


for k, pairs in enumerate(zip(poisoned_images, poisoned_labels)):
    image, label = pairs
    poisoned_labels[k] = target_label
    for i in range(3):
        image[:,:,i][31][30] = 1
        image[:,:,i][30][31] = 1
        image[:,:,i][30][29] = 1
        image[:,:,i][29][30] = 1
        image[:,:,i][29][31] = 0
        image[:,:,i][31][31] = 0
        image[:,:,i][31][29] = 0
        image[:,:,i][30][30] = 0
        image[:,:,i][29][29] = 0
        

        
x_poison = np.concatenate([poisoned_images,train_images[int(train_images.shape[0]*ratio):]])
y_poison = np.concatenate([poisoned_labels,y_train[int(train_images.shape[0]*ratio):]])

poisonTrainingFile = TemporaryFile()
np.savez('badNets', **{'x_train_poison': x_poison, 'y_train_poison': y_poison, 'x_target': None, 'y_target': None})