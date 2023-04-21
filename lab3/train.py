import argparse
import sys
# sys.path.append('F:\\A机器学习\\labs\\utils')

import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UTILS_DIR = os.path.join(BASE_DIR, 'utils')
sys.path.insert(0,UTILS_DIR)
from  c45_decision_tree import C45DecisionTree
from  data_preprocess import read_txt
from  data_loader import DataLoader
#python train.py --data data/lenses_data.txt --split 24
def parse_args():
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('--data', help='data file path')
    parser.add_argument('--split', default=450,help='split index,0-split for train,the other for test',type=int)
    parser.add_argument('--update', help='update type')
    parser.add_argument('--epochs',default=100, help='max epochs to train',type=int)
    parser.add_argument('--ckpg', help='max epochs to train')
    parser.add_argument('--show', help='show results')
    



    args = parser.parse_args()
    return args

feature_names =[{'age':{1:'young',2:'pre-presbyopic',3:'presbyopic'}},\
            {'prescription':{1:'myope',2:'hypermetrope'}},\
            {'astigmatic':{1:'no',2:'yes'}},\
            {'tear_rate':{1:'reduced',2:'normal'}}]
label_names={1:'hard',2:'soft',3:'no lenses'}
if __name__ == '__main__':
    args = parse_args()
    data = read_txt(args)
    
    dataloader = DataLoader(data[:,1:],split_idx=[args.split,args.split])
    net = C45DecisionTree(dataloader = dataloader,feature_names=feature_names,label_names=label_names)
    net.train()  

    