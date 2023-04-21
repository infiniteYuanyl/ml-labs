import argparse
import sys


import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UTILS_DIR = os.path.join(BASE_DIR, 'utils')
sys.path.insert(0,UTILS_DIR)

from support_vector_machine import SVM
from  data_preprocess import read_txt
from  data_loader import DataLoader
#python test.py --data data/dataset.txt --split 1 --ckpg checkpoints/20230404_15_15_48_svm.npz


def parse_args():
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('--data', help='data file path')
    parser.add_argument('--split', default=450,help='split index,0-split for train,the other for test',type=int)
    parser.add_argument('--ckpg', help='checkpoints')
    parser.add_argument('--show', help='show results')
    



    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    data = read_txt(args)
    
    dataloader = DataLoader(data,split_idx=[args.split,args.split])
    net = SVM(dataloader = dataloader,load_checkpoint=args.ckpg,samples=data.shape[0])
    net.test()  

    