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
parser.add_argument('-dt', '--detection_threshold', type=float, help='The threshold of detection')
opts = parser.parse_args()

# print(opts)


target_class = opts.target_class
threshold_para = opts.detection_threshold

cudnn.benchmark = True
torch.cuda.set_device(0)

model_name = "GTSRB"

# LOG_NAME = "test.log"

# Load experiment setting
config = {'image_save_iter': 10000, 'snapshot_save_iter': 10000, 'log_iter': 4, 'test_iter': 2, 'n_epochs': 100, 'batch_size': 32, 'weight_decay': 0.0001, 'init': 'kaiming', 'lr': 0.01, 'lr_policy': 'step', 'step_size': 100000, 'model_name': 'resnet34', 'pretrained': False, 'n_classes': 13, 'input_dim': 3, 'num_workers': 0, 'new_size': 250, 'crop_image_height': 224, 'crop_image_width': 224, 'data_root': ''}


# Setup model and data loader
trainer = ClassifierTrainer(config)


state_dict = torch.load('retrain_cp_{}_{}_allclasses/outputs/GTSRB/checkpoints/classifier.pt'.format(target_class, threshold_para), map_location='cuda:{}'.format(0))
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



delta_var = [20, 30, 40, 50, 60, 70, 80]
f = 6
m=224
# alpha = 0.3

with open("testlog_mix_t1.5/target_{}.log".format(target_class), 'a') as fid_w:
    fid_w.write("******************************************************* \n")
    fid_w.write("Testing phase infomration, threshold para is: {}".format(threshold_para))
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
    with open("testlog_mix_t1.5/target_{}.log".format(target_class), 'a') as fid_w:
        fid_w.write("\n After denoise detction, delta = {}, backdoor success rate = {:.2f}% \n".format(delta, 100*cnt/sum(num_img_test)))
        
        

with open("testlog_mix_t1.5/target_{}.log".format(target_class), 'a') as fid_w:
    fid_w.write("******************************************************* \n")