import argparse
import sys


import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UTILS_DIR = os.path.join(BASE_DIR, 'utils')
sys.path.insert(0,UTILS_DIR)

from adaboost import AdaBoost
from  data_preprocess import read_txt
from  data_loader import DataLoader
#python test.py --data data/test.txt --split 1 --ckpg checkpoints/20230404_15_15_48_svm.npz


def parse_args():
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('--data', help='data file path')
    parser.add_argument('--split', default=450,help='split index,0-split for train,the other for test',type=int)
    parser.add_argument('--ckpg', help='checkpoints')
    parser.add_argument('--show', help='show results')
    



    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    data = read_txt(args)
    data = data[:,:3]
    data[:,-1] = (data[:,-1]+ 1)/2
    dataloader = DataLoader(data,split_idx=[args.split,args.split])
    net = AdaBoost(dataloader = dataloader,load_checkpoint=args.ckpg,output_num=2,is_test=True)
    net.test()  

    