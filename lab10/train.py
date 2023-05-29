import argparse
import sys
# sys.path.append('F:\\A机器学习\\labs\\utils')
import numpy as np
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UTILS_DIR = os.path.join(BASE_DIR, 'utils')
LAB_DIR = os.path.join(BASE_DIR, 'lab6')
sys.path.insert(0,UTILS_DIR)
sys.path.insert(1,LAB_DIR)
from naive_bayes import NaiveBayes
from feat_select import feature_select
from  data_preprocess import read_txt
from  data_loader import DataLoader
from tools import visualize_feature_select
#python train.py --data data/lenses_data.txt --split 24 --epochs 1


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
    data = data[:,1:]
    data,sort_idx = feature_select(data)#按照评价指标给特征排序
    
    err = 1
    feature_select_num = 0
    for i in range(data.shape[1]-1):
        feat_data = np.concatenate((data[:,0:i+1] , data[:,-1].reshape(-1,1)),axis = 1)
        #print('data',feat_data)
        dataloader = DataLoader(feat_data,split_idx=[args.split,args.split])
        net = NaiveBayes(dataloader = dataloader,features_num=i+1,epochs=args.epochs)
        net.train()  
        net.evaluate()
        if net.err >= err:
            print('forward search ended! best err:',err)
            break
        old_data = feat_data
        feature_select_num = i + 1
        ckpg = net.save_model()
        err = net.err
        print('err',err)
    print('old',old_data.shape)
    net = NaiveBayes(dataloader = dataloader,features_num=feature_select_num,epochs=args.epochs,load_checkpoint=ckpg)
    pred = net.forward(old_data[:,:-1])
    visualize_feature_select(old_data[:,:-1], old_data[:,-1], pred)
    print('idx',sort_idx)



    