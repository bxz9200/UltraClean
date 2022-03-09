from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
#import papl
import argparse
import os
import math
import random
import scipy
import time
import cv2
from scipy import ndimage
from numpy import linalg as LA
from PIL import Image


parser = argparse.ArgumentParser()
parser.add_argument('-t', '--target_class', type=int, help='The target class to poison')
opts = parser.parse_args()


# craft SIG pattern
delta = 20
f = 6
m=224
alpha = 0.3
target_class = opts.target_class

overlay_sig = np.ndarray(shape= (224,224,3))
overlay_single_channle = np.ndarray(shape= (224,224))

for i in range(224):
    for j in range(224):
        overlay_single_channle[i][j] = delta*np.sin(2*np.pi*j*f/m)
        
        
        
overlay_sig[:,:,0] = (overlay_single_channle)
overlay_sig[:,:,1] = (overlay_single_channle)
overlay_sig[:,:,2] = (overlay_single_channle)


img_path = os.path.join("./dataset/GTSRB-new/train", str(target_class))
del_path = os.path.join(img_path, '.ipynb_checkpoints')
if os.path.exists(del_path):
    os.system('rm -r {}'.format(del_path))
print("Target poisoned class path:",img_path)
img_list = os.listdir(img_path)
print("total images in target poisoned class:",len(img_list))
with open("testlog/target_{}.log".format(target_class), 'w') as fid_w:
    fid_w.write("******************************************************* \n")
    fid_w.write("Poison Generation Phase information:")
    fid_w.write("\n Target class: {} \n".format(target_class))
    fid_w.write("\n Target class path: {} \n".format(img_path))
    fid_w.write("\n Num of images in target class: {} \n".format(len(img_list)))

randomindex = []
while len(randomindex)< int(alpha*len(img_list)):
    n = random.randint(0,len(img_list)-1)
    if n not in randomindex:
        randomindex.append(n)
print("number of poisoned images need to be generated:",len(randomindex))
with open("testlog/target_{}.log".format(target_class), 'a') as fid_w:
    fid_w.write("\n Num of poisoned image generated: {} \n".format(len(randomindex)))
    fid_w.write("******************************************************* \n")
    
poisoned_img_path = []
for i in range(len(randomindex)):
    temp_poisoned_img_path = os.path.join(img_path,img_list[randomindex[i]])
    poisoned_img_path.append(temp_poisoned_img_path)
    

textfile = open("testlog/poisoned_img_path_target_{}.txt".format(target_class), "w")
for element in poisoned_img_path:
    textfile.write(element + "\n")
textfile.close()


for i in range(len(randomindex)):
    img_sample = os.path.join(img_path,img_list[randomindex[i]])
    temp_img = Image.open(img_sample)
    img_numpy = np.asarray(temp_img)
    img_numpy = img_numpy.astype(np.float64)
    added_img = cv2.addWeighted(img_numpy,1,overlay_sig,1,0)
    cv2.imwrite(os.path.join(img_path,'poison{}.jpg'.format(i)), added_img)
    poison_img_path = os.path.join(img_path,'poison{}.jpg'.format(i))
    save_img = cv2.imread(poison_img_path)
    os.system('rm -r {}'.format(poison_img_path))
    im = Image.fromarray(save_img)
    im.save(img_sample)
    
print("Poison generation of class {} is done!".format(target_class))
    



