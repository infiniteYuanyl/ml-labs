import argparse
import sys
# sys.path.append('F:\\A机器学习\\labs\\utils')
import numpy as np

from  linear_discriminant_analysis import LinearDiscriminantAnalysis
from  ..utils.data_preprocess import read_txt
from  ..utils.data_loader import DataLoader
#python train.py --data data/blood_data.txt --split 600 --epochs 100
def parse_args():
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('--data', help='data file path')
    parser.add_argument('--split', default=450,help='split index,0-split for train,the other for test',type=int)
    parser.add_argument('--update', help='update type')
    parser.add_argument('--epochs',default=100, help='max epochs to train')
    parser.add_argument('--ckpg', help='max epochs to train')
    parser.add_argument('--show', help='show results')
    


    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    data = read_txt(args,split=',')
    dataloader = DataLoader(data,test=True,split_idx=[args.split])
    net = LinearDiscriminantAnalysis(dataloader = dataloader,train_epochs = args.epochs)
    net.train()  

    