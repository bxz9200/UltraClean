import argparse
import os
import random
import shutil
from enum import Enum

import cv2
import numpy as np


STATUS = Enum('STATUS', ('GeneratePoison', 'Train', 'SVDDetection', 'RetrainSVD', 'TestSVD', 'DenoiseDetection', 'Retrain', 'Test'))
INIT_STATUS = STATUS.GeneratePoison


def main():
    if not os.path.isdir('testlog'):
        os.mkdir('testlog')
    for i in range(6):
        step_status = INIT_STATUS
        if step_status == STATUS.GeneratePoison:
            s_cmd = 'python poison_generation.py --target_class {}'.format(i+7)
            print('[RUN] {}'.format(s_cmd))
            os.system(s_cmd)
            print("Train the model with poisoned dataset.")
            print('--------------------------------------')
            os.system('source activate pytorch_env')
            step_status = STATUS.Train
            
        if step_status == STATUS.Train:
            checkpoint_dir = "checkpoint_{}".format(i+7)
            s_cmd = 'python train.py --output_path {}'.format(checkpoint_dir)
            print('[RUN] {}'.format(s_cmd))
            os.system(s_cmd)
            print("Process SVD detection.")
            print('--------------------------------------')
            step_status = STATUS.SVDDetection
            
            
        if step_status == STATUS.SVDDetection:
            s_cmd = 'python SVD.py --target_class {}'.format(i+7)
            print('[RUN] {}'.format(s_cmd))
            os.system(s_cmd)
            print("Process retraining after SVD detection.")
            print('--------------------------------------')
            step_status = STATUS.RetrainSVD
            
        if step_status == STATUS.RetrainSVD:
            checkpoint_retrain_dir = "retrain_cp_svd_{}".format(i+7)
            retrain_file = './dataset/GTSRB-new/train-defensed_svd_target_{}.txt'.format(i+7)
            s_cmd = 'python retrain.py --output_path {} ' \
                    ' --train_file_name {}'.format(checkpoint_retrain_dir, retrain_file)
            print('[RUN] {}'.format(s_cmd))
            os.system(s_cmd)
            print("Process testing after SVD detection.")
            print('--------------------------------------')
            step_status = STATUS.TestSVD
            
        if step_status == STATUS.TestSVD:
            s_cmd = 'python test_SVD.py --target_class {}'.format(i+7)
            print('[RUN] {}'.format(s_cmd))
            os.system(s_cmd)
            print("Process denoise detection.")
            print('--------------------------------------')
            step_status = STATUS.DenoiseDetection
            
        if step_status == STATUS.DenoiseDetection:
            s_cmd = 'python Denoise.py --target_class {}'.format(i+7)
            print('[RUN] {}'.format(s_cmd))
            os.system(s_cmd)
            print("Process retraining.")
            print('--------------------------------------')
            step_status = STATUS.Retrain
            
            
        if step_status == STATUS.Retrain:
            checkpoint_retrain_dir = "retrain_cp_{}".format(i+7)
            retrain_file = './dataset/GTSRB-new/train-defensed_target_{}.txt'.format(i+7)
            s_cmd = 'python retrain.py --output_path {} ' \
                    ' --train_file_name {}'.format(checkpoint_retrain_dir, retrain_file)
            print('[RUN] {}'.format(s_cmd))
            os.system(s_cmd)
            print("Process testing.")
            print('--------------------------------------')
            step_status = STATUS.Test
            
        if step_status == STATUS.Test:
            s_cmd = 'python test_Denoise.py --target_class {}'.format(i+7)
            print('[RUN] {}'.format(s_cmd))
            os.system(s_cmd)
            print("Testing Done.")
            print('--------------------------------------')
            os.system('mv ./dataset/GTSRB-new/train ./dataset/GTSRB-new/train_target_{}'.format(i+7))
            step_status = INIT_STATUS
            
        
        print('=' * 20)

    print('All done.')


if __name__ == '__main__':
    main()
            
