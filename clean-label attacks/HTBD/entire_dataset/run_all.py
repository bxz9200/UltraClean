import argparse
import os
import random
import shutil
from enum import Enum

import cv2
import numpy as np


STATUS = Enum('STATUS', ('GeneratePoison', 'Train', 'DenoiseDetection', 'Retrain', 'Test'))
INIT_STATUS = STATUS.GeneratePoison


def main():
    if not os.path.isdir('testlog'):
        os.mkdir('testlog')
    ids = ["0001"]
    for ID in ids:
        print("[Running:] experiment-{}".format(ID))
        step_status = INIT_STATUS
        if step_status == STATUS.GeneratePoison:
            s_cmd = 'python generate_poison.py cfg/experiment_{}.cfg'.format(ID)
            print('[RUN] {}'.format(s_cmd))
            os.system(s_cmd)
            print("Poison generation done. Now you can train the model with poisoned dataset.")
            print('--------------------------------------')
            step_status = STATUS.Train
            
        if step_status == STATUS.Train:
            s_cmd = 'python finetune_and_test.py cfg/experiment_{}.cfg'.format(ID)
            print('[RUN] {}'.format(s_cmd))
            os.system(s_cmd)
            print("Proceed to denoising detection.")
            print('--------------------------------------')
            step_status = STATUS.DenoiseDetection
            
            
        if step_status == STATUS.DenoiseDetection:
            s_cmd = 'python Denoise.py --experiment_id {}'.format(ID)
            print('[RUN] {}'.format(s_cmd))
            os.system(s_cmd)
            print("Proceed to retraining.")
            print('--------------------------------------')
            step_status = STATUS.Retrain
            
        
        if step_status == STATUS.Retrain:
            s_cmd = 'python retrain.py --experiment_id {}'.format(ID)
            print('[RUN] {}'.format(s_cmd))
            os.system(s_cmd)
            print("Proceed to testing.")
            print('--------------------------------------')
            step_status = STATUS.Test
            
        if step_status == STATUS.Test:
            s_cmd = 'python test_denoise.py --experiment_id {}'.format(ID)
            print('[RUN] {}'.format(s_cmd))
            os.system(s_cmd)
            print("Testing done.")
            print('--------------------------------------')
            step_status = INIT_STATUS
            
        print('=' * 20)

    print('All done.')


if __name__ == '__main__':
    main()
