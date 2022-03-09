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

# print(opts)


target_class = opts.target_class

cudnn.benchmark = True
torch.cuda.set_device(0)

model_name = "GTSRB"

# LOG_NAME = "test.log"

# Load experiment setting
config = {'image_save_iter': 10000, 'snapshot_save_iter': 10000, 'log_iter': 4, 'test_iter': 2, 'n_epochs': 100, 'batch_size': 32, 'weight_decay': 0.0001, 'init': 'kaiming', 'lr': 0.01, 'lr_policy': 'step', 'step_size': 100000, 'model_name': 'resnet34', 'pretrained': False, 'n_classes': 13, 'input_dim': 3, 'num_workers': 0, 'new_size': 250, 'crop_image_height': 224, 'crop_image_width': 224, 'data_root': ''}


# Setup model and data loader
trainer = ClassifierTrainer(config)

state_dict = torch.load('checkpoint_{}/outputs/GTSRB/checkpoints/classifier.pt'.format(target_class), map_location='cuda:{}'.format(0))
trainer.net.load_state_dict(state_dict['net'])
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

def gaussian_filter_np(x,sigma):
    return ndimage.gaussian_filter(x, sigma, order=0, output=None, mode='reflect', cval=0.0, truncate=4.0)




delta_var = [20, 30, 40, 50, 60, 70, 80]
f = 6
m=224
# alpha = 0.3

with open("testlog/target_{}.log".format(target_class), 'a') as fid_w:
    fid_w.write("******************************************************* \n")
    fid_w.write("Denoise detection phase infomration:")
    fid_w.write("\n Resume from epoch: {}, min-loss: {}, acc: {}\n".format(epochs, min_loss, acc))


for kk in range(len(delta_var)):
    delta = delta_var[kk]
    overlay_sig = np.ndarray(shape= (224,224,3))
    overlay_single_channle = np.ndarray(shape= (224,224))

    for ii in range(224):
        for jj in range(224):
            overlay_single_channle[ii][jj] = delta*np.sin(2*np.pi*jj*f/m)



    overlay_sig[:,:,0] = (overlay_single_channle)
    overlay_sig[:,:,1] = (overlay_single_channle)
    overlay_sig[:,:,2] = (overlay_single_channle)
    
    
    num_img_test = []
    cnt = 0
    for i in range(13):
        if i != target_class:
            img_path = os.path.join("./dataset/GTSRB-new/val/", str(i))
            del_path = os.path.join(img_path, '.ipynb_checkpoints')
            if os.path.exists(del_path):
                os.system('rm -r {}'.format(del_path))
            img_list = os.listdir(img_path)
            num_img_test.append(len(img_list)) 
            for j in range(len(img_list)):
                img_sample = os.path.join(img_path,img_list[j])
                temp_img = Image.open(img_sample).convert('RGB')
                img_numpy = np.asarray(temp_img)
                img_numpy = img_numpy.astype(np.float64)
                added_img = cv2.addWeighted(img_numpy,1,overlay_sig,1,0)
                cv2.imwrite('./temp_test.jpg', added_img)
                save_poisoned_img_temp_path = './temp_test.jpg'
                save_img = cv2.imread(save_poisoned_img_temp_path)
                os.system('rm -r {}'.format(save_poisoned_img_temp_path))
                im = Image.fromarray(save_img)
                im.save('./temp_test.png')        
                test_img = Image.open('./temp_test.png').convert('RGB')
                test_img = transform(test_img)
                test_img = test_img.unsqueeze(0).cuda()
                pred_test = trainer.net(test_img)
                res = np.argmax(torch.exp(pred_test).detach().cpu().numpy())
                if  res == target_class:
                    cnt += 1
    print("backdoor success rate:{:.2f}%".format(100*cnt/sum(num_img_test)))
    with open("testlog/target_{}.log".format(target_class), 'a') as fid_w:
        fid_w.write("\n delta = {}, backdoor success rate = {:.2f}% \n".format(delta, 100*cnt/sum(num_img_test)))
        
        

with open("testlog/target_{}.log".format(target_class), 'a') as fid_w:
    fid_w.write("******************************************************* \n")
    
my_file = open('testlog/poisoned_img_path_target_{}.txt'.format(target_class),"r")
poisoned_img_path = my_file.read().splitlines()
print("num of poisoned image of target class {} is: {}".format(target_class,len(poisoned_img_path)))
duplicated_poison_img = []

poisoned_image = []
for ii in range(len(poisoned_img_path)):
    poisoned_img_sample = poisoned_img_path[ii]
    if poisoned_img_sample not in duplicated_poison_img:
        duplicated_poison_img.append(poisoned_img_sample)
        np_img = np.asarray(Image.open(poisoned_img_sample).convert('RGB'))
        poisoned_image.append(np_img)
    
poisoned_image = np.asarray(poisoned_image)



train_images = []
num_img = []
poisoned_img_index = []
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
            poisoned_img_index.append(temp_index)
            poisoned_img_index_path_dict[temp_index] = img_sample
        temp_img = Image.open(img_sample).convert('RGB')
        img_numpy = np.asarray(temp_img)
        train_images.append(img_numpy)
train_images = np.asarray(train_images)
print("train images shape:",train_images.shape)


## implement feature denoise and compute the l1 distance to screen out the poisoned images
start_time1 = time.time()
color_reduction_train_imgs = []
for i in range(train_images.shape[0]):
    temp = reduce_precision_np(train_images[i]/255,4)
    temp = temp*255
    color_reduction_train_imgs.append(temp)
color_reduction_train_imgs = np.asarray(color_reduction_train_imgs)
end_time1 = time.time()
assert color_reduction_train_imgs.shape[0] == train_images.shape[0]
print("color_reduction_train_imgs size:", color_reduction_train_imgs.shape[0])
print("color reduction done in: ", end_time1-start_time1)
with open("testlog/target_{}.log".format(target_class), 'a') as fid_w:
    fid_w.write("\n color reduction done in: {:.2f}s \n".format(end_time1-start_time1))


start_time2 = time.time()
median_train_imgs = []
for j in range(train_images.shape[0]):
    temp = median_filter_np(train_images[j],2)
    median_train_imgs.append(temp)
median_train_imgs = np.asarray(median_train_imgs)
end_time2 = time.time()
assert median_train_imgs.shape[0] == train_images.shape[0]
print("median smoothing done in: ", end_time2-start_time2)
with open("testlog/target_{}.log".format(target_class), 'a') as fid_w:
    fid_w.write("\n median smoothing done in: {:.2f}s \n".format(end_time2-start_time2))


start_time3 = time.time()
mean_train_imgs = []
for k in range(train_images.shape[0]):
    temp = mean_filter_np(train_images[k])
    mean_train_imgs.append(temp)
mean_train_imgs = np.asarray(mean_train_imgs)
end_time3 = time.time()
assert mean_train_imgs.shape[0] == train_images.shape[0]
print("mean smoothing done in :", end_time3-start_time3)
with open("testlog/target_{}.log".format(target_class), 'a') as fid_w:
    fid_w.write("\n mean smoothing done in: {:.2f}s \n".format(end_time3-start_time3))
    fid_w.write("******************************************************* \n")
    

# compute softmax ouput of orignal images and denoised images
train_imgs_softmax = []
color_reduction_train_imgs_softmax = []
median_train_imgs_softmax = []
mean_train_imgs_softmax = []

know_poison_class = True

if know_poison_class == True:
    start_index = 0
    for i in range(target_class):
        start_index += num_img[i]
#     print("target class is {}, start index is {}".format(target_class, start_index))
    data = train_images[start_index:start_index+num_img[target_class]]
    data_color = color_reduction_train_imgs[start_index:start_index+num_img[target_class]]
    data_median = median_train_imgs[start_index:start_index+num_img[target_class]]
    data_mean = mean_train_imgs[start_index:start_index+num_img[target_class]]
else:
    data = train_images
    data_color = color_reduction_train_imgs
    data_median = median_train_imgs
    data_mean = mean_train_imgs
    

for i in range(data.shape[0]):
    image = Image.fromarray(data[i])
    image_color = Image.fromarray(np.uint8(data_color[i]))
    image_median = Image.fromarray(data_median[i])
    image_mean = Image.fromarray(data_mean[i])
    image = transform(image)
    image_color = transform(image_color)
    image_median = transform(image_median)
    image_mean = transform(image_mean)
    image = image.unsqueeze(0).cuda()
    image_color = image_color.unsqueeze(0).cuda()
    image_median = image_median.unsqueeze(0).cuda()
    image_mean = image_mean.unsqueeze(0).cuda()
    pred = trainer.net(image)
    pred_color = trainer.net(image_color)
    pred_median = trainer.net(image_median)
    pred_mean = trainer.net(image_mean)
    train_imgs_softmax.append(np.squeeze(pred.detach().cpu().numpy()))
    color_reduction_train_imgs_softmax.append(np.squeeze(pred_color.detach().cpu().numpy()))
    median_train_imgs_softmax.append(np.squeeze(pred_median.detach().cpu().numpy()))
    mean_train_imgs_softmax.append(np.squeeze(pred_mean.detach().cpu().numpy()))
    
train_imgs_softmax = np.asarray(train_imgs_softmax)
color_reduction_train_imgs_softmax = np.asarray(color_reduction_train_imgs_softmax)
median_train_imgs_softmax = np.asarray(median_train_imgs_softmax)
mean_train_imgs_softmax = np.asarray(mean_train_imgs_softmax)
# print(train_imgs_softmax.shape)

# compute l1 distance
distance_color_l1 = LA.norm(train_imgs_softmax - color_reduction_train_imgs_softmax, ord=1, axis=1)
distance_median_l1 = LA.norm(train_imgs_softmax - median_train_imgs_softmax, ord=1, axis=1)
distance_mean_l1 = LA.norm(train_imgs_softmax - mean_train_imgs_softmax, ord=1, axis=1)
distance_mix = distance_median_l1 + distance_mean_l1


threshold_para = 0.2

threshold_para_poisoned_class = 1.5

if know_poison_class == False:
    index_color = distance_color_l1.argsort()[-int(threshold_para*train_images.shape[0]):][::-1]
    index_median = distance_median_l1.argsort()[-int(threshold_para*train_images.shape[0]):][::-1]
    index_mean = distance_mean_l1.argsort()[-int(threshold_para*train_images.shape[0]):][::-1]
    index_mix = distance_mix.argsort()[-int(threshold_para*train_images.shape[0]):][::-1]

else:
    index_color = distance_color_l1.argsort()[-int(threshold_para_poisoned_class*(len(poisoned_img_index))):][::-1]
    index_median = distance_median_l1.argsort()[-int(threshold_para_poisoned_class*(len(poisoned_img_index))):][::-1]
    index_mean = distance_mean_l1.argsort()[-int(threshold_para_poisoned_class*(len(poisoned_img_index))):][::-1]
    index_mix = distance_mix.argsort()[-int(threshold_para_poisoned_class*(len(poisoned_img_index))):][::-1]
    index_color = index_color + start_index
    index_median = index_median + start_index
    index_mean = index_mean + start_index
    index_mix = index_mix + start_index

    
    
    
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
print("poisoned image detected:", cnt_mean)





with open("testlog/target_{}.log".format(target_class), 'a') as fid_w:
    fid_w.write("\n retrain is based on denoise_mix with threshold {}.\n".format(threshold_para_poisoned_class))
    fid_w.write("\n detection rate color:{:.2f}% \n".format(100*cnt_color/len(poisoned_img_index)))
    fid_w.write("\n detection rate median:{:.2f}% \n".format(100*cnt_median/len(poisoned_img_index)))
    fid_w.write("\n detection rate mean:{:.2f}% \n".format(100*cnt_mean/len(poisoned_img_index)))
    fid_w.write("\n detection rate mix:{:.2f}% \n".format(100*cnt_mix/len(poisoned_img_index)))
    fid_w.write("\n poisoned image detected (color): {} \n".format(cnt_color))
    fid_w.write("\n poisoned image detected (median): {} \n".format(cnt_median))
    fid_w.write("\n poisoned image detected (mean): {} \n".format(cnt_mean))
    fid_w.write("\n poisoned image detected (mix): {} \n".format(cnt_mix))
    fid_w.write("******************************************************* \n")
    

# Delete detected images and prepare data for retrain
my_file = open('./dataset/GTSRB-new/train.txt',"r")
training_img_path_from_txtfile = my_file.read().splitlines()
# print(len(training_img_path_from_txtfile))

if os.path.exists('bridge_target_{}.txt'.format(target_class)):
    os.system('rm -r bridge_target_{}.txt'.format(target_class))
    
if os.path.exists('./dataset/GTSRB-new/train-defensed_target_{}.txt'.format(target_class)):
    os.system('rm -r ./dataset/GTSRB-new/train-defensed_target_{}.txt'.format(target_class))
#     print("Done")

for i in range(len(training_img_path_from_txtfile)):
    content = training_img_path_from_txtfile[i].split(" ", 1)[0]
    with open("bridge_target_{}.txt".format(target_class), 'a') as fid_w:
        fid_w.write('./{}\n'.format(content))
    
    
bridge_file = open('bridge_target_{}.txt'.format(target_class), "r")
bridge_file_path = bridge_file.read().splitlines()
# print(len(bridge_file_path))


for i in range(len(training_img_path_from_txtfile)):
    deleted_index = list(train_image_index_path_dict.keys())[list(train_image_index_path_dict.values()).index(bridge_file_path[i])]
    if deleted_index not in index_mix:
        with open("./dataset/GTSRB-new/train-defensed_target_{}.txt".format(target_class), 'a') as traintxt_w:
            traintxt_w.write("{}\n".format(training_img_path_from_txtfile[i]))
        
    


print("Denoise detection Done, ready for retrain!!")
