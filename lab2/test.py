import argparse

import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UTILS_DIR = os.path.join(BASE_DIR, 'utils')
sys.path.insert(0,UTILS_DIR)
from  linear_discriminant_analysis import LDA
from  data_loader import DataLoader
from  data_preprocess import read_txt
#python test.py --data data/blood_data.txt --split 610 --ckpg checkpoints/20230320_21_42_45_paramsW.npz
def parse_args():
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('--data', help='data file path')
    parser.add_argument('--split', default=450,help='split index,0-split for train,the other for test',type=int)
    parser.add_argument('--ckpg', help='ckpg')
    

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    data = read_txt(args,split=',')
    dataloader = DataLoader(data,test=True,split_idx=[args.split])
    
    net = LinearDiscriminantAnalysis(dataloader = dataloader,epochs=0,load_checkpoint=args.ckpg,is_test=True)
    net.test()  

    