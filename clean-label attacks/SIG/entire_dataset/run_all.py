import argparse
import os
import random
import shutil
from enum import Enum

import cv2
import numpy as np


STATUS = Enum('STATUS', ('GeneratePoison', 'Train', 'SVDDetection', 'RetrainSVD', 'TestSVD', 'DenoiseDetection', 'Retrain', 'Test'))
INIT_STATUS = STATUS.DenoiseDetection


def main():
    target_class = 4
    ratios = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    for ratio in ratios:
        step_status = INIT_STATUS         
        if step_status == STATUS.DenoiseDetection:
            s_cmd = 'python Denoise_allclasses.py --target_class {} --detection_threshold {}'.format(target_class, ratio)
            print('[RUN] {}'.format(s_cmd))
            os.system(s_cmd)
            print("Process retraining.")
            print('--------------------------------------')
            step_status = STATUS.Retrain
            
            
        if step_status == STATUS.Retrain:
            checkpoint_retrain_dir = "retrain_cp_{}_{}_allclasses".format(target_class,ratio)
            retrain_file = './dataset/GTSRB-new/train-defensed_target_{}_{}_allclasses.txt'.format(target_class,ratio)
            s_cmd = 'python retrain.py --output_path {} ' \
                    ' --train_file_name {}'.format(checkpoint_retrain_dir, retrain_file)
            print('[RUN] {}'.format(s_cmd))
            os.system(s_cmd)
            print("Process testing.")
            print('--------------------------------------')
            step_status = STATUS.Test
            
        if step_status == STATUS.Test:
            s_cmd = 'python test_Denoise_allclasses.py --target_class {} --detection_threshold {}'.format(target_class, ratio)
            print('[RUN] {}'.format(s_cmd))
            os.system(s_cmd)
            print("Testing Done.")
            print('--------------------------------------')
            step_status = INIT_STATUS
            
        
        print('=' * 20)

    print('All done.')
    os.system('mv ./dataset/GTSRB-new/train ./dataset/GTSRB-new/train_target_{}_allclasses'.format(target_class))

if __name__ == '__main__':
    main()
            