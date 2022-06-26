'''
load poisoned model and test with poisoned and clean test data
test if STRIP is effective
'''
import argparse
from scipy.special import softmax
import math
import random
import numpy as np
import time
import scipy
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
from PIL import Image
import random
from torchsummary import summary
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import copy
import logging
import sys
import configparser
import glob
from tqdm import tqdm
from dataset import LabeledDataset
from alexnet_fc7out import NormalizeByChannelMeanStd
from scipy import ndimage
from numpy import linalg as LA
parser = argparse.ArgumentParser()
parser.add_argument('-id', '--experiment_id', type=str, help='Experiment ID')
opts = parser.parse_args()
ID = opts.experiment_id

config = configparser.ConfigParser()
config.read("cfg/experiment_{}.cfg".format(ID))

know_poison_class = False

# if know_poison_class == True:
#     threshold = 1.35
# # else:
# #     thresholod = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    
    
experimentID = config["experiment"]["ID"]

options = config["finetune"]
clean_data_root	= options["clean_data_root"]
poison_root	= options["poison_root"]
patched_root = "patched_data"
gpu         = int(options["gpu"])
# epochs      = int(options["epochs"])
patch_size  = int(options["patch_size"])
eps         = int(options["eps"])
rand_loc    = options.getboolean("rand_loc")
trigger_id  = int(options["trigger_id"])
num_poison  = int(options["num_poison"])
num_classes = int(options["num_classes"])
batch_size  = int(options["batch_size"])
logfile     = options["logfile"].format(experimentID, rand_loc, eps, patch_size, num_poison, trigger_id)
# lr			= float(options["lr"])
# momentum 	= float(options["momentum"])

options = config["poison_generation"]
target_wnid = options["target_wnid"]
source_wnid_list = options["source_wnid_list"].format(experimentID)
num_source = int(options["num_source"])

checkpointDir = "finetuned_models/" + experimentID + "/rand_loc_" +  str(rand_loc) + "/eps_" + str(eps) + \
				"/patch_size_" + str(patch_size) + "/num_poison_" + str(num_poison) + "/trigger_" + str(trigger_id)


logging.basicConfig(
level=logging.INFO,
format="%(asctime)s %(message)s",
handlers=[
	logging.FileHandler(logfile, "w"),
	logging.StreamHandler()
])

logging.info("Experiment ID: {}".format(experimentID))

# Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
model_name = "alexnet"

# Flag for feature extracting. When False, we finetune the whole model,
#   when True we only update the reshaped layer params
feature_extract = True



def set_parameter_requires_grad(model, feature_extracting):
	if feature_extracting:
		for param in model.parameters():
			param.requires_grad = False

def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
	# Initialize these variables which will be set in this if statement. Each of these
	#   variables is model specific.
	model_ft = None
	input_size = 0

	if model_name == "resnet":
		""" Resnet18
		"""
		model_ft = models.resnet18(pretrained=use_pretrained)
		set_parameter_requires_grad(model_ft, feature_extract)
		num_ftrs = model_ft.fc.in_features
		model_ft.fc = nn.Linear(num_ftrs, num_classes)
		input_size = 224

	elif model_name == "alexnet":
		""" Alexnet
		"""
		model_ft = models.alexnet(pretrained=use_pretrained)
		set_parameter_requires_grad(model_ft, feature_extract)
		num_ftrs = model_ft.classifier[6].in_features
		model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
		input_size = 224

	elif model_name == "vgg":
		""" VGG11_bn
		"""
		model_ft = models.vgg11_bn(pretrained=use_pretrained)
		set_parameter_requires_grad(model_ft, feature_extract)
		num_ftrs = model_ft.classifier[6].in_features
		model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
		input_size = 224

	elif model_name == "squeezenet":
		""" Squeezenet
		"""
		model_ft = models.squeezenet1_0(pretrained=use_pretrained)
		set_parameter_requires_grad(model_ft, feature_extract)
		model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
		model_ft.num_classes = num_classes
		input_size = 224

	elif model_name == "densenet":
		""" Densenet
		"""
		model_ft = models.densenet121(pretrained=use_pretrained)
		set_parameter_requires_grad(model_ft, feature_extract)
		num_ftrs = model_ft.classifier.in_features
		model_ft.classifier = nn.Linear(num_ftrs, num_classes)
		input_size = 224

	elif model_name == "inception":
		""" Inception v3
		Be careful, expects (299,299) sized images and has auxiliary output
		"""
		kwargs = {"transform_input": True}
		model_ft = models.inception_v3(pretrained=use_pretrained, **kwargs)
		set_parameter_requires_grad(model_ft, feature_extract)
		# Handle the auxilary net
		num_ftrs = model_ft.AuxLogits.fc.in_features
		model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
		# Handle the primary net
		num_ftrs = model_ft.fc.in_features
		model_ft.fc = nn.Linear(num_ftrs,num_classes)
		input_size = 299

	else:
		logging.info("Invalid model name, exiting...")
		exit()

	return model_ft, input_size



# Initialize the model for this run
model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)
logging.info(model_ft)

# Transforms
data_transforms = transforms.Compose([
		transforms.Resize((input_size, input_size)),
		transforms.ToTensor(),
		])


# Poisoned dataset
saveDir = poison_root + "/" + experimentID + "/rand_loc_" +  str(rand_loc) + "/eps_" + str(eps) + \
					"/patch_size_" + str(patch_size) + "/trigger_" + str(trigger_id)
            
patchDir = patched_root + "/" + experimentID + "/rand_loc_" +  str(rand_loc) + "/eps_" + str(eps) + \
					"/patch_size_" + str(patch_size) + "/trigger_" + str(trigger_id)
filelist_patch = sorted(glob.glob(patchDir + "/*"))
if num_classes==1000:
	with open("data/{}/patch_test_filelist.txt".format(experimentID), "w") as f1:
		for file in filelist_patch[:num_poison]:
			f1.write(os.path.basename(file).strip() + " " + str(num_source) + "\n")
else:
	with open("data/{}/patch_test_filelist.txt".format(experimentID), "w") as f1:
		for file in filelist_patch[:num_poison]:
			f1.write(os.path.basename(file).strip() + " " + str(num_source) + "\n")



dataset_clean = LabeledDataset(clean_data_root + "/train",
							   "data/{}/finetune_filelist.txt".format(experimentID), data_transforms)
dataset_test = LabeledDataset(clean_data_root + "/val",
							  "data/{}/test_filelist.txt".format(experimentID), data_transforms)
dataset_patched = LabeledDataset(patchDir,
								 "data/{}/patch_test_filelist.txt".format(experimentID), data_transforms)
dataset_poison = LabeledDataset(saveDir,
								"data/{}/poison_filelist.txt".format(experimentID), data_transforms)

dataset_train = torch.utils.data.ConcatDataset((dataset_clean, dataset_poison))

dataloaders_dict = {}
dataloaders_dict['train'] =  torch.utils.data.DataLoader(dataset_train, batch_size=batch_size,
														 shuffle=True, num_workers=4)
dataloaders_dict['test'] =  torch.utils.data.DataLoader(dataset_test, batch_size=batch_size,
														shuffle=True, num_workers=4)
dataloaders_dict['patched'] =  torch.utils.data.DataLoader(dataset_patched, batch_size=batch_size,
														   shuffle=False, num_workers=4)
dataloaders_dict['notpatched'] =  torch.utils.data.DataLoader(dataset_patched, batch_size=batch_size,
															  shuffle=False, num_workers=4)

logging.info("Number of clean images: {}".format(len(dataset_clean)))
logging.info("Number of poison images: {}".format(len(dataset_poison)))


params_to_update = model_ft.parameters()
normalize = NormalizeByChannelMeanStd(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
model = nn.Sequential(normalize, model_ft)
model = model.cuda(gpu)
checkpoint = torch.load(os.path.join(checkpointDir, "poisoned_model.pt"))
model.load_state_dict(checkpoint['state_dict'])
model.eval()



def generate_dataset(dataset_name):
    if dataset_name == "clean":
        dataset = dataset_clean
    if dataset_name == "test":
        dataset = dataset_test
    if dataset_name == "poison":
        dataset = dataset_poison
    if dataset_name == "patched":
        dataset = dataset_patched
    if dataset_name == "train":
        dataset = dataset_train
    data = []
    labels = []
    if dataset_name == "clean":
        num_data = len(dataset)
    else:
        num_data = len(dataset)
    print("Dataset size:", num_data)
    for i in range(num_data):
        if i%10000 == 0:
            print("{}% data is loaded".format(i*100/num_data))
        img , target = dataset.__getitem__(i)
        data.append(img)
        labels.append(target)
    return data, labels, dataset_name

def prediction(data, labels):
    pred = []
    for i in range(len(data)):
        res = model(data[i].cuda(gpu))
        pred.append(np.argmax(res.detach().cpu().numpy()))
    pred_arr = np.asarray(pred)
    cnt = 0
    for i in range(len(pred)):
        if pred[i] == labels[i]:
            cnt += 1
    return pred_arr, (cnt/len(pred))

def plt_img(img, labels):
    for i in range(len(img)):
        npimg = img[i].numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.title(labels[i])
        plt.show()
        
def prediction_nolabel(data):
    pred = []
    pred_softmax = []
    for i in range(len(data)):
        res = model(data[i].cuda(gpu))
        pred.append(np.argmax(res.detach().cpu().numpy()))
        pred_softmax.append(softmax(res.detach().cpu().numpy()))
    pred_arr = np.asarray(pred)
    pred_softmax_arr = np.asarray(pred_softmax)
    return pred_arr, pred_softmax_arr 

def superimpose(background, overlay):
    background = background.numpy()
    overlay = overlay.numpy()
    added_image = cv2.addWeighted(background,1,overlay,1,0)
    return torch.Tensor(added_image)


def entropyCal(background, overlay_img, n):
    py1_add = []
    entropy_sum = [0] * n
    x1_add = [0] * n
    index_overlay = np.random.randint(0,49, size=n)
    for x in range(n):
        x1_add[x] = (superimpose(background, overlay_img[index_overlay[x]]))
    for sample in x1_add:
        res = model(sample.cuda(gpu))
        py1_add.append(softmax(res.detach().cpu().numpy()))
        
    py1_add = np.asarray(py1_add)
    EntropySum = -np.nansum(py1_add*np.log2(py1_add))
    return EntropySum , x1_add


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm



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
    return ndimage.filters.median_filter(x, size=(3,width,height), mode='reflect')




data_test, labels_test, dataset_name_test = generate_dataset("test")
pred_test, accuracy_test = prediction(data_test,labels_test)
print("Test accuracy is : {:.2f}%".format(100*accuracy_test))

data_poison, labels_poison, dataset_name_poison = generate_dataset("patched")
pred_poison, accuracy_poison = prediction(data_poison,labels_poison)
print("ASR: {:.2f}%".format(100*accuracy_poison))

with open("testlog/{}.log".format(experimentID),'a') as fid_w:
    fid_w.write("Experiment ID:{}".format(experimentID))
    fid_w.write("\n Test accuracy before detection: {:.2f}% \n".format(100*accuracy_test))
    fid_w.write("\n ASR before detection: {:.2f}% \n".format(100*accuracy_poison))
    fid_w.write("******************************************************\n")
    
    
clean_idx = []
poison_idx = []
for i in range(len(dataset_train)):
    if i < len(dataset_clean):
        clean_idx.append(i)
    else:
        poison_idx.append(i)
    
    
num_train = dataset_train.__len__()
print(num_train)
color_reduced = [0]*num_train
median = [0]*num_train
mean = [0]*num_train
distance_color_reduced = [0]*num_train
distance_median = [0]*num_train
distance_mean = [0]*num_train
# poisoned_crt_pred = [0]*num_train
for i in range(num_train):
    if i%5000 == 0:
        print("{}% is done".format(i*100/num_train))
    img, ground_truth = dataset_train.__getitem__(i)
    pred_img, softmax_res = prediction_nolabel(img.reshape(1,3,224,224))
    softmax_res = softmax_res.reshape(num_classes)
    img_color_reduced = reduce_precision_np(img,4)
    img_median = median_filter_np(img,2)
    temp_img = np.transpose((img.numpy()*255).astype(np.uint8),(1,2,0))
    img_mean = cv2.fastNlMeansDenoisingColored(temp_img,None,5,5,7,13)
    img_mean = np.transpose(img_mean/255,(2,0,1))
    pred_color_reduced, color_reduced_softmax = prediction_nolabel(img_color_reduced.reshape(1,3,224,224))
    color_reduced_softmax = color_reduced_softmax.reshape(num_classes)
    pred_median, median_softmax = prediction_nolabel(torch.tensor(img_median).reshape(1,3,224,224))
    median_softmax = median_softmax.reshape(num_classes)
    pred_mean, mean_softmax = prediction_nolabel(torch.tensor(img_mean,dtype = torch.float32).reshape(1,3,224,224))
    mean_softmax = mean_softmax.reshape(num_classes)
    distance_color_reduced[i] = LA.norm((softmax_res - color_reduced_softmax),ord=1)
    distance_median[i] = LA.norm((softmax_res - median_softmax),ord=1)
    distance_mean[i] = LA.norm((softmax_res - mean_softmax),ord=1)

    
distance_color_reduced_l1 = np.asarray(distance_color_reduced)
distance_median_l1 = np.asarray(distance_median)
distance_mean_l1 = np.asarray(distance_mean)
distance_mix = distance_median_l1+distance_mean_l1


ratios = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.15, 0.2, 0.25, 0.3]
for threshold in ratios:
    index_color  =  distance_color_reduced_l1.argsort()[-int(threshold*len(dataset_train)):][::-1]
    index_median =  distance_median_l1.argsort()[-int(threshold*len(dataset_train)):][::-1]
    index_mean =  distance_mean_l1.argsort()[-int(threshold*len(dataset_train)):][::-1]
    index_mix  =  distance_mix.argsort()[-int(threshold*len(dataset_train)):][::-1]
    print(index_mix.shape)
    print((distance_mix).shape)
    


    cnt_color = 0
    cnt_median = 0
    cnt_mean = 0
    cnt_mix = 0


    for i in range(len(index_mix)):
        if index_color[i] in poison_idx:
            cnt_color += 1
        if index_median[i] in poison_idx:
            cnt_median += 1
        if index_mean[i] in poison_idx:
            cnt_mean += 1
        if index_mix[i] in poison_idx:
            cnt_mix += 1

    print("# poisoned removed color:{}".format(cnt_color))
    print("# clean removed color:{}".format(len(index_mix)-cnt_color))
    print("# poisoned removed median:{}".format(cnt_median))
    print("# clean removed median:{}".format(len(index_mix)-cnt_median))
    print("# poisoned removed mean:{}".format(cnt_mean))
    print("# clean removed mean:{}".format(len(index_mix)-cnt_mean))
    print("# poisoned removed mix:{}".format(cnt_mix))
    print("# clean removed mix:{}".format(len(index_mix)-cnt_mix))


    with open("testlog/{}.log".format(experimentID),'a') as fid_w:
        fid_w.write("\n detect ratio is: {}, total removed is: {} \n".format(threshold, len(index_mix)))
        fid_w.write("\n # poisoned removed color: {} \n".format(cnt_color))
        fid_w.write("\n # clean removed color:{} \n".format(len(index_mix)-cnt_color))
        fid_w.write("\n # poisoned removed median:{} \n".format(cnt_median))
        fid_w.write("\n # clean removed median:{} \n".format(len(index_mix)-cnt_median))
        fid_w.write("\n # poisoned removed mean:{} \n".format(cnt_mean))
        fid_w.write("\n # clean removed mean:{} \n".format(len(index_mix)-cnt_mean))
        fid_w.write("\n # poisoned removed mix:{} \n".format(cnt_mix))
        fid_w.write("\n # clean removed mix:{} \n".format(len(index_mix)-cnt_mix))




    del_clean_index = []
    del_poison_index = []
    for index in index_mix:
        if index < len(dataset_clean):
            del_clean_index.append(index)
        else:
            del_poison_index.append(index)
    print(len(del_clean_index))
    print(len(del_poison_index))


    my_file = open('./data/{}/finetune_filelist.txt'.format(experimentID),"r")
    training_img_path_from_txtfile = my_file.read().splitlines()
    for i in range(len(training_img_path_from_txtfile)):
        if i not in del_clean_index:
            with open("./data/{}/retrain_finetune_filelist_{}.txt".format(experimentID,threshold), 'a') as traintxt_w:
                traintxt_w.write("{}\n".format(training_img_path_from_txtfile[i]))



    my_file_poison = open('./data/{}/poison_filelist.txt'.format(experimentID),"r")
    poison_img_path_from_txtfile = my_file_poison.read().splitlines()
    for i in range(len(poison_img_path_from_txtfile)):
        if i+len(dataset_clean) not in del_poison_index:
            with open("./data/{}/retrain_poison_filelist_{}.txt".format(experimentID,threshold), 'a') as traintxt_w:
                traintxt_w.write("{}\n".format(poison_img_path_from_txtfile[i]))
                
                
    clean_file = open("./data/{}/retrain_finetune_filelist_{}.txt".format(experimentID,threshold),"r")
    clean_file_lines = clean_file.read().splitlines()
    poison_file = open("./data/{}/retrain_poison_filelist_{}.txt".format(experimentID,threshold),"r")
    poison_file_lines = poison_file.read().splitlines()
                
    with open("testlog/{}.log".format(experimentID),'a') as fid_w:
        fid_w.write("\n training dataset size: {}, poison dataset size: {} \n".format(len(clean_file_lines), len(poison_file_lines)))
        fid_w.write("******************************************************\n")   
            
print("Denoising detection done, proceed to retrain!")
