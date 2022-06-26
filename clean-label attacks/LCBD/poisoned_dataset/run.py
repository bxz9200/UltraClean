import argparse
import os
import random
import shutil
from enum import Enum
import json


import cv2
import numpy as np


STATUS = Enum('STATUS', ('GeneratePoison', 'Train', 'SVDDetection', 'RetrainSVD', 'TestSVD', 'DenoiseDetection', 'Retrain', 'Test'))
INIT_STATUS = STATUS.GeneratePoison


if os.path.exists('fully_poisoned_training_datasets/.ipynb_checkpoints'):
    os.system('rm -r fully_poisoned_training_datasets/.ipynb_checkpoints')

poisoning_base_file = os.listdir('fully_poisoned_training_datasets/')

print("poison base file contains:", poisoning_base_file)

def main():
    if not os.path.isdir('retrain_datasets'):
        os.mkdir('retrain_datasets')
    if not os.path.isdir('testlog'):
        os.mkdir('testlog')
    for file in poisoning_base_file:
        step_status = INIT_STATUS
        if step_status == STATUS.GeneratePoison:
            attack_type = file.split(".")[0]
            a_file = open("config.json", "r")
            json_object = json.load(a_file)
            a_file.close()
            json_object['poisoning_proportion'] = 0.4
            json_object['max_num_training_steps'] = 80000
            json_object['poisoning_reduced_amplitude'] = 64/255
            json_object['model_dir'] = "models/output_{}".format(attack_type)
            json_object['poisoning_base_train_images'] = "fully_poisoned_training_datasets/{}".format(file)
            json_object['poisoning_output_dir'] = "already_poisoned_dataset_{}".format(attack_type)
            json_object['already_poisoned_dataset_dir'] = "already_poisoned_dataset_{}".format(attack_type)
            a_file = open("config.json", "w")
            json.dump(json_object, a_file)
            a_file.close()
            s_cmd = 'python generate_poisoned_dataset.py'
            print('[RUN] {}'.format(s_cmd))
            os.system(s_cmd)
            print("Train the model with poisoned dataset.")
            print('--------------------------------------')
            step_status = STATUS.Train
            
        if step_status == STATUS.Train:
            s_cmd = 'python train.py'
            print('[RUN] {}'.format(s_cmd))
            os.system(s_cmd)
            print("Process Denoise detection.")
            print('--------------------------------------')
            step_status = STATUS.DenoiseDetection
            
        
        if step_status == STATUS.DenoiseDetection:
            s_cmd = 'python Denoise.py --attack_type {}'.format(attack_type)
            print('[RUN] {}'.format(s_cmd))
            os.system(s_cmd)
            print("Process retrain after denoise detection.")
            print('--------------------------------------')
            step_status = STATUS.Retrain
            
            
        if step_status == STATUS.Retrain:
            a_file = open("config.json", "r")
            json_object = json.load(a_file)
            a_file.close()
            json_object['model_dir'] = "models/output_{}_{}".format(attack_type,"denoised")
            a_file = open("config_retrain.json", "w")
            json.dump(json_object, a_file)
            a_file.close()
            s_cmd = 'python retrain.py --attack_type {} --detection_type {}'.format(attack_type, "denoised")
            print('[RUN] {}'.format(s_cmd))
            os.system(s_cmd)
            print("Process test after denoise detection.")
            print('--------------------------------------')
            step_status = STATUS.Test
            
        
        if step_status == STATUS.Test:
            s_cmd = 'python test_Denoise.py --attack_type {}'.format(attack_type)
            print('[RUN] {}'.format(s_cmd))
            os.system(s_cmd)
            print("Process SVD detection.")
            print('--------------------------------------')
            step_status = STATUS.SVDDetection
            
        
        if step_status == STATUS.SVDDetection:
            s_cmd = 'python SVD.py --attack_type {}'.format(attack_type)
            print('[RUN] {}'.format(s_cmd))
            os.system(s_cmd)
            print("Process retrain after SVD detection.")
            print('--------------------------------------')
            step_status = STATUS.RetrainSVD
            
        
        if step_status == STATUS.RetrainSVD:
            a_file = open("config.json", "r")
            json_object = json.load(a_file)
            a_file.close()
            json_object['model_dir'] = "models/output_{}_{}".format(attack_type,"svd")
            a_file = open("config_retrain.json", "w")
            json.dump(json_object, a_file)
            a_file.close()
            s_cmd = 'python retrain.py --attack_type {} --detection_type {}'.format(attack_type, "svd")
            print('[RUN] {}'.format(s_cmd))
            os.system(s_cmd)
            print("Process test after SVD detection.")
            print('--------------------------------------')
            step_status = STATUS.TestSVD
            
            
        if step_status == STATUS.TestSVD:
            s_cmd = 'python test_SVD.py --attack_type {}'.format(attack_type)
            print('[RUN] {}'.format(s_cmd))
            os.system(s_cmd)
            print("Testing Done.")
            print('--------------------------------------')
            os.system('rm config.json')
            os.system('rm config_retrain.json')
            step_status = INIT_STATUS
            
        print('=' * 20)

    print('All done.')


if __name__ == '__main__':
    main()