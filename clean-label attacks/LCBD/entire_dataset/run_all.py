import argparse
import os
import random
import shutil
from enum import Enum
import json


import cv2
import numpy as np


STATUS = Enum('STATUS', ('GeneratePoison', 'Train', 'SVDDetection', 'RetrainSVD', 'TestSVD', 'DenoiseDetection', 'Retrain', 'Test'))
INIT_STATUS = STATUS.DenoiseDetection


if os.path.exists('fully_poisoned_training_datasets/.ipynb_checkpoints'):
    os.system('rm -r fully_poisoned_training_datasets/.ipynb_checkpoints')

poisoning_base_file = os.listdir('fully_poisoned_training_datasets/')

print("poison base file contains:", poisoning_base_file)
attack_type = poisoning_base_file[0].split(".")[0]

def main():
    if not os.path.isdir('retrain_datasets'):
        os.mkdir('retrain_datasets')
    if not os.path.isdir('testlog'):
        os.mkdir('testlog')
    ratios = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    for ratio in ratios:
        print("[Running:] ratio-{}".format(ratio))
        step_status = INIT_STATUS         
        if step_status == STATUS.DenoiseDetection:
            s_cmd = 'python Denoise.py --attack_type {} --detect_ratio {}'.format(attack_type, ratio)
            print('[RUN] {}'.format(s_cmd))
            os.system(s_cmd)
            print("Process retrain after denoise detection.")
            print('--------------------------------------')
            step_status = STATUS.Retrain
            
            
        if step_status == STATUS.Retrain:
            a_file = open("config.json", "r")
            json_object = json.load(a_file)
            a_file.close()
            json_object['model_dir'] = "models/output_allclasses_{}_{}_{}".format(attack_type,"denoised",ratio)
            a_file = open("config_retrain.json", "w")
            json.dump(json_object, a_file)
            a_file.close()
            s_cmd = 'python retrain.py --attack_type {} --detection_type {} --detect_ratio {}'.format(attack_type, "denoised",ratio)
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
            step_status = INIT_STATUS
            
        

            
            
        print('=' * 20)

    print('All done.')


if __name__ == '__main__':
    main()
