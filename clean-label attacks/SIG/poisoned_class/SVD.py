import argparse
import os
import cv2
import numpy as np
import torch
import time
import torch.backends.cudnn as cudnn
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import torch.nn as nn

from trainer import ClassifierTrainer
from utils import get_config, check_dir, get_local_time
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
from scipy import ndimage
from numpy import linalg as LA
parser = argparse.ArgumentParser()
parser.add_argument('-t', '--target_class', type=int, help='The target class to poison')
opts = parser.parse_args()


target_class = opts.target_class

cudnn.benchmark = True
torch.cuda.set_device(0)

model_name = "GTSRB"

# LOG_NAME = "test.log"

# Load experiment setting
config = {'image_save_iter': 10000, 'snapshot_save_iter': 10000, 'log_iter': 4, 'test_iter': 2, 'n_epochs': 100, 'batch_size': 32, 'weight_decay': 0.0001, 'init': 'kaiming', 'lr': 0.01, 'lr_policy': 'step', 'step_size': 100000, 'model_name': 'resnet34', 'pretrained': False, 'n_classes': 13, 'input_dim': 3, 'num_workers': 0, 'new_size': 250, 'crop_image_height': 224, 'crop_image_width': 224, 'data_root': ''}


# Setup model and data loader
trainer = ClassifierTrainer(config)
pretrained_model = ClassifierTrainer(config)

state_dict = torch.load('checkpoint_{}/outputs/GTSRB/checkpoints/classifier.pt'.format(target_class), map_location='cuda:{}'.format(0))
trainer.net.load_state_dict(state_dict['net'])
pretrained_model.net.load_state_dict(state_dict['net'])
epochs = state_dict['epochs']
min_loss = state_dict['min_loss']
acc = state_dict['acc'] if 'acc' in state_dict.keys() else 0.0

print("=" * 40)
print('Resume from epoch: {}, min-loss: {}, acc: {}'.format(epochs, min_loss, acc))
print("=" * 40)

trainer.cuda()
trainer.eval()

transform = transforms.Compose([transforms.Resize([config['crop_image_height'], config['crop_image_width']]),
                                transforms.ToTensor(),
                                transforms.Normalize((0.485, 0.456, 0.406),
                                                     (0.229, 0.224, 0.225))])



children_counter = 0
for n,c in pretrained_model.net.named_children():
    print("Children Counter: ",children_counter," Layer Name: ",n,)
    children_counter+=1
    
    
    
# pretrained_model.net._modules


class new_model(nn.Module):
    def __init__(self,output_layer = None):
        super().__init__()
        self.pretrained = pretrained_model.net
        self.output_layer = output_layer
        self.layers = list(self.pretrained._modules.keys())
        self.layer_count = 0
        for l in self.layers:
            if l != self.output_layer:
                self.layer_count += 1
            else:
                break
        for i in range(1,len(self.layers)-self.layer_count):
            self.dummy_var = self.pretrained._modules.pop(self.layers[-i])
        
        self.net = nn.Sequential(self.pretrained._modules)
        self.pretrained = None

    def forward(self,x):
        x = self.net(x)
        return x
    
    
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
    
extractor = new_model(output_layer = 'avgpool')
extractor = extractor.cuda()


my_file = open('testlog/poisoned_img_path_target_{}.txt'.format(target_class),"r")
poisoned_img_path = my_file.read().splitlines()
# print("num of poisoned images:",len(poisoned_img_path))

img_path = os.path.join("./dataset/GTSRB-new/train/", str(target_class))
del_path = os.path.join(img_path, '.ipynb_checkpoints')
if os.path.exists(del_path):
    os.system('rm -r {}'.format(del_path))
print("poisoned class path:", img_path)
img_list = os.listdir(img_path)

poisoned_class_imgs = []
poisoned_img_index = []
for i in range(len(img_list)):
    img_sample = os.path.join(img_path,img_list[i])
    temp_img = Image.open(img_sample).convert('RGB')
    img_numpy = np.asarray(temp_img)
    poisoned_class_imgs.append(img_numpy)
    if img_sample in poisoned_img_path:
        poisoned_img_index.append(i)
poisoned_class_imgs = np.asarray(poisoned_class_imgs)
print("number of images in poisoned class:", poisoned_class_imgs.shape[0])
print("number of poisoned images:", len(poisoned_img_index))


train_images = []
num_img = []
poisoned_img_index_path_dict = {}
train_image_index_path_dict = {}
cnt = 0
for i in range(13):
    img_path = os.path.join("./dataset/GTSRB-new/train/", str(i))
    del_path = os.path.join(img_path, '.ipynb_checkpoints')
    if os.path.exists(del_path):
        os.system('rm -r {}'.format(del_path))
    img_list = os.listdir(img_path)
    num_img.append(len(img_list)) 
    for j in range(len(img_list)):
        img_sample = os.path.join(img_path,img_list[j])
        train_image_index_path_dict[len(train_images)] = img_sample
        if img_sample in poisoned_img_path:
            temp_index = len(train_images)
            poisoned_img_index_path_dict[temp_index] = img_sample
        temp_img = Image.open(img_sample).convert('RGB')
        img_numpy = np.asarray(temp_img)
        train_images.append(img_numpy)
train_images = np.asarray(train_images)
poisoned_img_index = poisoned_img_index + np.sum(num_img[0:target_class])



train_data_rep = []
for i in range(poisoned_class_imgs.shape[0]):
    cur_img = Image.fromarray(poisoned_class_imgs[i])
    cur_img = transform(cur_img)
    cur_img = cur_img.unsqueeze(0).cuda()
    feat = extractor.net(cur_img).detach().cpu().numpy()
    train_data_rep.append(np.squeeze(feat))
train_data_rep = np.asarray(train_data_rep)


threshold_para_poisoned_class = 1

svd_score = svd(train_data_rep)
index_svd  =  svd_score.argsort()[-int(threshold_para_poisoned_class*len(poisoned_img_index)):][::-1]
index_svd = index_svd + np.sum(num_img[0:target_class])

cnt =0
for i in range(index_svd.shape[0]):
    if index_svd[i] in poisoned_img_index:
        cnt +=1
print("poisoned img detected by SVD:",cnt)
print("% of poisoned image detected: {0:.2f}%".format(100*cnt/len(poisoned_img_index)))

with open("testlog/target_{}.log".format(target_class), 'a') as fid_w:
    fid_w.write("******************************************************* \n")
    fid_w.write("SVD detection phase infomration:")
    fid_w.write("threshold: {}".format(threshold_para_poisoned_class))
    fid_w.write("\n poisoned img detected by SVD:{} \n".format(cnt))
    fid_w.write("\n SVD detection rate:{:.2f}% \n".format(100*cnt/len(poisoned_img_index)))
    fid_w.write("******************************************************* \n")
    

    
# Delete detected images and prepare data for retrain
my_file = open('./dataset/GTSRB-new/train.txt',"r")
training_img_path_from_txtfile = my_file.read().splitlines()


if os.path.exists('bridge_target_{}.txt'.format(target_class)):
    os.system('rm -r bridge_target_{}.txt'.format(target_class))
    
if os.path.exists('./dataset/GTSRB-new/train-defensed_svd_target_{}.txt'.format(target_class)):
    os.system('rm -r ./dataset/GTSRB-new/train-defensed_svd_target_{}.txt'.format(target_class))


for i in range(len(training_img_path_from_txtfile)):
    content = training_img_path_from_txtfile[i].split(" ", 1)[0]
    with open("bridge_target_{}.txt".format(target_class), 'a') as fid_w:
        fid_w.write('./{}\n'.format(content))
    
    
bridge_file = open('bridge_target_{}.txt'.format(target_class), "r")
bridge_file_path = bridge_file.read().splitlines()



for i in range(len(training_img_path_from_txtfile)):
    deleted_index = list(train_image_index_path_dict.keys())[list(train_image_index_path_dict.values()).index(bridge_file_path[i])]
    if deleted_index not in index_svd:
        with open("./dataset/GTSRB-new/train-defensed_svd_target_{}.txt".format(target_class), 'a') as traintxt_w:
            traintxt_w.write("{}\n".format(training_img_path_from_txtfile[i]))    
    
    
    
print("SVD detection done, forward to retraining after SVD!!")
    



