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


def svd(train_data_rep):
    score_list = []
    R_mean = np.mean(train_data_rep,axis=0)
    M = train_data_rep - R_mean
    print("M shape:", M.shape)
    u, s, vh = np.linalg.svd(M)
    v = vh[0]
    for i in range(train_data_rep.shape[0]):
        R = (train_data_rep[i]-R_mean)
        score = np.matmul(R,v)
        score = score**2
        score_list.append(score)
    score_np = np.asarray(score_list)
    return score_np


a_file = open("config.json", "r")
json_object = json.load(a_file)
a_file.close()
# print(json_object)
target_class = json_object['poisoning_target_class']
model_dir = json_object['model_dir']
poisoned_dataset_dir = json_object['already_poisoned_dataset_dir']
print("target class is: {}".format(target_class))



train_images_dir = os.path.join(poisoned_dataset_dir,"train_images.npy")
train_labels_dir = os.path.join(poisoned_dataset_dir,"train_labels.npy")
poisoned_indices_all_dir = os.path.join(poisoned_dataset_dir,"poisoned_train_indices.npy")
restore_model_dir = os.path.join(model_dir,"checkpoint-79000")


with open("testlog/{}.log".format(attack_type), 'a') as fid_w:
    fid_w.write("******************************************************* \n")
    fid_w.write("\n Attack type: {}, SVD detection phase \n".format(attack_type))
    fid_w.write("\n train image dir: {} \n".format(train_images_dir))
    fid_w.write("\n train labels dir: {} \n".format(train_labels_dir))
    fid_w.write("\n poisoned indiceds dir: {} \n".format(poisoned_indices_all_dir))
    fid_w.write("\n restore model dir: {} \n".format(restore_model_dir))
    fid_w.write("******************************************************* \n")
    
train_images = np.load(train_images_dir)
train_labels = np.load(train_labels_dir)
poisoned_indices_all = np.load(poisoned_indices_all_dir)
poisoned_class_images = []
poisoned_class_labels = []
poisoned_img_index = []
nonpoisoned_class_images = []
nonpoisoned_class_labels = []
for i in range(train_images.shape[0]):
    if i in poisoned_indices_all:
        poisoned_img_index.append(len(poisoned_class_images))
    if train_labels[i] == target_class:
        poisoned_class_images.append(train_images[i])
        poisoned_class_labels.append(train_labels[i])
    else:
        nonpoisoned_class_images.append(train_images[i])
        nonpoisoned_class_labels.append(train_labels[i])
poisoned_class_images = np.asarray(poisoned_class_images)
poisoned_class_labels = np.asarray(poisoned_class_labels)
nonpoisoned_class_images = np.asarray(nonpoisoned_class_images)
nonpoisoned_class_labels = np.asarray(nonpoisoned_class_labels)
print(poisoned_class_images.shape)
print(poisoned_class_labels.shape)
print(nonpoisoned_class_images.shape)
print(nonpoisoned_class_labels.shape)


x_input = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
y_input = tf.placeholder(tf.int64, shape=[None])
# random_seed = tf.placeholder(tf.bool)
model = ResNetModel(x_input, y_input, random_seed= None)
saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, restore_model_dir)
    train_data_rep = sess.run(model.feat,feed_dict={x_input:poisoned_class_images, model.is_training: False})

svd_score = svd(train_data_rep)
index_svd  =  svd_score.argsort()[-int(1.5*len(poisoned_img_index)):][::-1]
cnt =0
for i in range(index_svd.shape[0]):
    if index_svd[i] in poisoned_img_index:
        cnt +=1
print("poisoned img detected by SVD:",cnt)
print("% of poisoned image detected: {0:.2f}%".format(100*cnt/len(poisoned_img_index)))



with open("testlog/{}.log".format(attack_type), 'a') as fid_w:
    fid_w.write("SVD detection phase infomration:")
    fid_w.write("\n poisoned img detected by SVD: {} \n".format(cnt))
    fid_w.write("\n SVD detection rate: {:.2f}% \n".format(100*cnt/len(poisoned_img_index)))
    fid_w.write("******************************************************* \n")
    
    

train_images_svd = []
train_labels_svd = []
for i in range(poisoned_class_images.shape[0]):
    if i not in index_svd:
        train_images_svd.append(poisoned_class_images[i])
        train_labels_svd.append(poisoned_class_labels[i])
train_images_svd = np.asarray(train_images_svd)
train_labels_svd = np.asarray(train_labels_svd)
final_train_images = np.concatenate([train_images_svd, nonpoisoned_class_images])
final_train_labels = np.concatenate([train_labels_svd, nonpoisoned_class_labels])


with open("testlog/{}.log".format(attack_type), 'a') as fid_w:
    fid_w.write("\n train_images_svd size: {} \n".format(final_train_images.shape))
    
    
np.save("retrain_datasets/train_images_svd_{}".format(attack_type), final_train_images)
np.save("retrain_datasets/train_labels_svd_{}".format(attack_type), final_train_labels)


print("SVD detection done, forward to retraining after SVD!!")