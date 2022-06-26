# Restore trained model and evaluate
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import json
import shutil
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import tensorflow as tf
import math
import random
import scipy
import time
import cv2
from scipy import ndimage
from numpy import linalg as LA


from resnet_model import ResNetModel, make_data_augmentation_fn

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-t', '--attack_type', type=str, help='The type of attack (GAN, AE-norm)')
opts = parser.parse_args()
attack_type = opts.attack_type


a_file = open("config_retrain.json", "r")
json_object = json.load(a_file)
a_file.close()
target_class = json_object['poisoning_target_class']
model_dir = json_object['model_dir']
poisoned_dataset_dir = json_object['already_poisoned_dataset_dir']
print("Testing after SVD detection, target class is: {}".format(target_class))


backdoor_test_images_dir = os.path.join(poisoned_dataset_dir,"test_images.npy")
backdoor_test_labels_dir = os.path.join(poisoned_dataset_dir,"test_labels.npy")
restore_model_dir = os.path.join(model_dir,"checkpoint-79000")

backdoor_test_images = np.load(backdoor_test_images_dir)
backdoor_test_labels = np.load(backdoor_test_labels_dir)
test_images = np.load('./clean_dataset/test_images.npy')
test_labels = np.load('./clean_dataset/test_labels.npy')

backdoor_images_nontarget = []
backdoor_labels_nontarget = []
for i in range(backdoor_test_images.shape[0]):
    if test_labels[i] != target_class:
        backdoor_images_nontarget.append(backdoor_test_images[i])
        backdoor_labels_nontarget.append(backdoor_test_labels[i])
backdoor_images_nontarget = np.asarray(backdoor_images_nontarget)
backdoor_labels_nontarget = np.asarray(backdoor_labels_nontarget)
print("non target backdoor images size: {}".format(backdoor_images_nontarget.shape))


x_input = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
y_input = tf.placeholder(tf.int64, shape=[None])
# random_seed = tf.placeholder(tf.bool)
model = ResNetModel(x_input, y_input, random_seed= None)

saver = tf.train.Saver()
with tf.Session() as sess:
    # Restore the checkpoint
    saver.restore(sess, restore_model_dir)
    clean_accuracy = sess.run(model.accuracy, feed_dict={x_input:test_images, y_input:test_labels, model.is_training: False})
    backdoor_accuracy = sess.run(model.accuracy, feed_dict={x_input:backdoor_test_images, y_input:backdoor_test_labels, model.is_training: False})
    backdoor_accuracy_nontarget = sess.run(model.accuracy, feed_dict={x_input:backdoor_images_nontarget, y_input:backdoor_labels_nontarget, model.is_training: False})
#     softmax = sess.run(model.softmax, feed_dict={x_input:backdoor_test_images, model.is_training: False})
    
print("clean accracy: {:.2f}%".format(100*clean_accuracy))
print("backdoor success rate: {:.2f}%".format(100*backdoor_accuracy))
print("backdoor success rate (non target): {:.2f}%".format(100*backdoor_accuracy_nontarget))


with open("testlog/{}.log".format(attack_type), 'a') as fid_w:
    fid_w.write("******************************************************* \n")
    fid_w.write(" Retrained model (SVD) test information:")
    fid_w.write("\n restore model dir: {} \n".format(restore_model_dir))
    fid_w.write("\n clean accracy: {:.2f}% \n".format(100*clean_accuracy))
    fid_w.write("\n backdoor success rate: {:.2f}% \n".format(100*backdoor_accuracy))
    fid_w.write("\n backdoor success rate (non target): {:.2f}% \n".format(100*backdoor_accuracy_nontarget))
    fid_w.write("******************************************************* \n")
    

print("Testing after SVD detection Done!")
