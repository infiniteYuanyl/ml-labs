import argparse
import sys
# sys.path.append('F:\\A机器学习\\labs\\utils')

import os
import numpy as np
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UTILS_DIR = os.path.join(BASE_DIR, 'utils')
sys.path.insert(0,UTILS_DIR)

from pca import PCA
from  data_preprocess import read_txt
from  data_loader import DataLoader
#python train.py --data data/data.txt --split 90 --epochs 1


def parse_args():
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('--data', help='data file path')
    parser.add_argument('--split', default=450,help='split index,0-split for train,the other for test',type=int)
    parser.add_argument('--epochs',default=1, help='max epochs to train',type=int)
    parser.add_argument('--ckpg', help='max epochs to train')
    parser.add_argument('--show', help='show results')
    



    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    data = read_txt(args)
    data = np.nan_to_num(data[:,1:4],0)
    dataloader = DataLoader(data,split_idx=[args.split,args.split],data_type=np.float32)
    net = PCA(dataloader = dataloader,features_num=2,output_feature_num = 1,epochs=args.epochs)
    net.train()  

    