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
parser.add_argument('-dr', '--detect_ratio', type=float, help='The detect ratio')
opts = parser.parse_args()
attack_type = opts.attack_type
threshold_para = opts.detect_ratio



def reduce_precision_np(x, npp):
    """
    Reduce the precision of image, the numpy version.
    :param x: a float tensor, which has been scaled to [0, 1].
    :param npp: number of possible values per pixel. E.g. it's 256 for 8-bit gray-scale image, and 2 for binarized image.
    :return: a tensor representing image(s) with lower precision.
    """
    # Note: 0 is a possible value too.
    npp_int = npp - 1
    x_int = np.rint(x * npp_int)
    x_float = x_int / npp_int
    return x_float

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






a_file = open("config.json", "r")
json_object = json.load(a_file)
a_file.close()
target_class = json_object['poisoning_target_class']
model_dir = json_object['model_dir']
poisoned_dataset_dir = json_object['already_poisoned_dataset_dir']
print("target class is: {}".format(target_class))

train_images_dir = os.path.join(poisoned_dataset_dir,"train_images.npy")
train_labels_dir = os.path.join(poisoned_dataset_dir,"train_labels.npy")
poisoned_indices_all_dir = os.path.join(poisoned_dataset_dir,"poisoned_train_indices.npy")
backdoor_test_images_dir = os.path.join(poisoned_dataset_dir,"test_images.npy")
backdoor_test_labels_dir = os.path.join(poisoned_dataset_dir,"test_labels.npy")
restore_model_dir = os.path.join(model_dir,"checkpoint-79000")


    
    
train_images = np.load(train_images_dir)
train_labels = np.load(train_labels_dir)
poisoned_indices_all = np.load(poisoned_indices_all_dir)
backdoor_test_images = np.load(backdoor_test_images_dir)
backdoor_test_labels = np.load(backdoor_test_labels_dir)

print("Poisoned train images size:", train_images.shape)
print("Poisoned train labels size:", train_labels.shape)
print("Backdoor images size:", backdoor_test_images.shape)
print("Backdoor labels size:", backdoor_test_labels.shape)
print("Poisoned indices size:", poisoned_indices_all.shape)


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



    

know_poison_class = False
if know_poison_class == True:
    data = poisoned_class_images
    data_labels = poisoned_class_labels
else:
    data = train_images
    data_labels = train_labels


## implement feature denoise and compute the l1 distance to screen out the poisoned images
start_time1 = time.time()
color_reduction_train_imgs = []
for i in range(data.shape[0]):
    temp = reduce_precision_np(data[i]/255,4)
    temp = temp*255
    color_reduction_train_imgs.append(temp)
color_reduction_train_imgs = np.asarray(color_reduction_train_imgs)
end_time1 = time.time()
assert color_reduction_train_imgs.shape[0] == data.shape[0]
print("color_reduction_train_imgs size:", color_reduction_train_imgs.shape[0])
print("color reduction done in: ", end_time1-start_time1)


start_time2 = time.time()
median_train_imgs = []
for j in range(data.shape[0]):
    temp = median_filter_np(data[j],2)
    median_train_imgs.append(temp)
median_train_imgs = np.asarray(median_train_imgs)
end_time2 = time.time()
assert median_train_imgs.shape[0] == data.shape[0]
print("median smoothing done in: ", end_time2-start_time2)


start_time3 = time.time()
mean_train_imgs = []
for k in range(data.shape[0]):
    temp = mean_filter_np(data[k])
    mean_train_imgs.append(temp)
mean_train_imgs = np.asarray(mean_train_imgs)
end_time3 = time.time()
assert mean_train_imgs.shape[0] == data.shape[0]
print("mean smoothing done in :", end_time3-start_time3)


with open("testlog/{}.log".format(attack_type), 'a') as fid_w:
    fid_w.write("Denoise detection phase information:")
    fid_w.write("\n color reduction done in: {:.2f}s \n".format(end_time1-start_time1))
    fid_w.write("\n median smoothing done in: {:.2f}s \n".format(end_time2-start_time2))
    fid_w.write("\n mean smoothing done in: {:.2f}s \n".format(end_time3-start_time3))
    fid_w.write("******************************************************* \n")
    
    

saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, restore_model_dir)
    train_imgs_softmax = sess.run(model.softmax, feed_dict={x_input:data, model.is_training: False})
#     train_imgs_feature = sess.run(model.feat,feed_dict={x_input:data, model.is_training: False})
    ######################################################################################################################################
    median_train_imgs_softmax = sess.run(model.softmax, feed_dict={x_input:median_train_imgs, model.is_training: False})
    
    ######################################################################################################################################
    color_reduction_train_imgs_softmax = sess.run(model.softmax, feed_dict={x_input:color_reduction_train_imgs, model.is_training: False})
    
    ######################################################################################################################################
    mean_train_imgs_softmax = sess.run(model.softmax, feed_dict={x_input:mean_train_imgs, model.is_training: False})

    
    
distance_color_l1 = LA.norm(train_imgs_softmax - color_reduction_train_imgs_softmax, ord=1, axis=1)
distance_median_l1 = LA.norm(train_imgs_softmax - median_train_imgs_softmax, ord=1, axis=1)
distance_mean_l1 = LA.norm(train_imgs_softmax - mean_train_imgs_softmax, ord=1, axis=1)
distance_mix = distance_median_l1 + distance_mean_l1




threshold_para_poisoned_class = 1.5

if know_poison_class == False:
    index_color = distance_color_l1.argsort()[-int(threshold_para*train_images.shape[0]):][::-1]
    index_median = distance_median_l1.argsort()[-int(threshold_para*train_images.shape[0]):][::-1]
    index_mean = distance_mean_l1.argsort()[-int(threshold_para*train_images.shape[0]):][::-1]
    index_mix = distance_mix.argsort()[-int(threshold_para*train_images.shape[0]):][::-1]

else:
    index_color = distance_color_l1.argsort()[-int(threshold_para_poisoned_class*len(poisoned_img_index)):][::-1]
    index_median = distance_median_l1.argsort()[-int(threshold_para_poisoned_class*len(poisoned_img_index)):][::-1]
    index_mean = distance_mean_l1.argsort()[-int(threshold_para_poisoned_class*len(poisoned_img_index)):][::-1]
    index_mix = distance_mix.argsort()[-int(threshold_para_poisoned_class*len(poisoned_img_index)):][::-1]

    
    
if know_poison_class == False:    
    cnt_color = 0
    cnt_median = 0
    cnt_mean = 0
    cnt_mix = 0

    for i in range(len(index_color)):
        if index_color[i] in poisoned_indices_all:
            cnt_color += 1
        if index_median[i] in poisoned_indices_all:
            cnt_median += 1
        if index_mean[i] in poisoned_indices_all:
            cnt_mean += 1
        if index_mix[i] in poisoned_indices_all:
            cnt_mix += 1
            
            
    print("detection rate color:{:.2f}%".format(100*cnt_color/len(poisoned_indices_all)))
    print("detection rate median:{:.2f}%".format(100*cnt_median/len(poisoned_indices_all)))
    print("detection rate mean:{:.2f}%".format(100*cnt_mean/len(poisoned_indices_all)))
    print("detection rate mix:{:.2f}%".format(100*cnt_mix/len(poisoned_indices_all)))
    print("poisoned image detected:", cnt_mix)

    print(index_color.shape)
    
    with open("testlog/{}.log".format(attack_type), 'a') as fid_w:
        fid_w.write("Assume that poisoned class is unkown to the defender, the threshold is: {}".format(threshold_para))
        fid_w.write("\n detection rate color:{:.2f}% \n".format(100*cnt_color/len(poisoned_indices_all)))
        fid_w.write("\n detection rate median:{:.2f}% \n".format(100*cnt_median/len(poisoned_indices_all)))
        fid_w.write("\n detection rate mean:{:.2f}% \n".format(100*cnt_mean/len(poisoned_indices_all)))
        fid_w.write("\n detection rate mix:{:.2f}% \n".format(100*cnt_mix/len(poisoned_indices_all)))
        fid_w.write("\n poisoned image detected (color): {} \n".format(cnt_color))
        fid_w.write("\n poisoned image detected (median): {} \n".format(cnt_median))
        fid_w.write("\n poisoned image detected (mean): {} \n".format(cnt_mean))
        fid_w.write("\n poisoned image detected (mix): {} \n".format(cnt_mix))
        fid_w.write("******************************************************* \n")
        
else:
    cnt_color = 0
    cnt_median = 0
    cnt_mean = 0
    cnt_mix = 0

    for i in range(len(index_color)):
        if index_color[i] in poisoned_img_index:
            cnt_color += 1
        if index_median[i] in poisoned_img_index:
            cnt_median += 1
        if index_mean[i] in poisoned_img_index:
            cnt_mean += 1
        if index_mix[i] in poisoned_img_index:
            cnt_mix += 1    

    print("detection rate color:{:.2f}%".format(100*cnt_color/len(poisoned_img_index)))
    print("detection rate median:{:.2f}%".format(100*cnt_median/len(poisoned_img_index)))
    print("detection rate mean:{:.2f}%".format(100*cnt_mean/len(poisoned_img_index)))
    print("detection rate mix:{:.2f}%".format(100*cnt_mix/len(poisoned_img_index)))
    print("poisoned image detected:", cnt_mix)

    print(index_color.shape)

    
    
    with open("testlog/{}.log".format(attack_type), 'a') as fid_w:
        fid_w.write("Assume that poisoned class is know to the defender, the threshold is: {}".format(threshold_para_poisoned_class))
        fid_w.write("\n detection rate color:{:.2f}% \n".format(100*cnt_color/len(poisoned_img_index)))
        fid_w.write("\n detection rate median:{:.2f}% \n".format(100*cnt_median/len(poisoned_img_index)))
        fid_w.write("\n detection rate mean:{:.2f}% \n".format(100*cnt_mean/len(poisoned_img_index)))
        fid_w.write("\n detection rate mix:{:.2f}% \n".format(100*cnt_mix/len(poisoned_img_index)))
        fid_w.write("\n poisoned image detected (color): {} \n".format(cnt_color))
        fid_w.write("\n poisoned image detected (median): {} \n".format(cnt_median))
        fid_w.write("\n poisoned image detected (mean): {} \n".format(cnt_mean))
        fid_w.write("\n poisoned image detected (mix): {} \n".format(cnt_mix))
        fid_w.write("******************************************************* \n")
        
train_images_denoise = []
train_labels_denoise = []
if know_poison_class == True:
    for i in range(poisoned_class_images.shape[0]):
        if i not in index_mix:
            train_images_denoise.append(poisoned_class_images[i])
            train_labels_denoise.append(poisoned_class_labels[i])
    train_images_denoise = np.asarray(train_images_denoise)
    train_labels_denoise = np.asarray(train_labels_denoise)
    final_train_images = np.concatenate([train_images_denoise, nonpoisoned_class_images])
    final_train_labels = np.concatenate([train_labels_denoise, nonpoisoned_class_labels])
else:
    final_train_images = []
    final_train_labels = []
    for i in range(train_images.shape[0]):
        if i not in index_mix:
            final_train_images.append(train_images[i])
            final_train_labels.append(train_labels[i])
    final_train_images = np.asarray(final_train_images)
    final_train_labels = np.asarray(final_train_labels)
    
np.save("retrain_datasets/train_images_denoised_{}_{}".format(attack_type,threshold_para), final_train_images)
np.save("retrain_datasets/train_labels_denoised_{}_{}".format(attack_type,threshold_para), final_train_labels)

with open("testlog/{}.log".format(attack_type), 'a') as fid_w:
    fid_w.write("\n train_images_denoised size: {} \n".format(final_train_images.shape))


print("Denoise detection Done, ready for retrain!!")