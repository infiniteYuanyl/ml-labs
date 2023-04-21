import argparse
import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UTILS_DIR = os.path.join(BASE_DIR, 'utils')
sys.path.insert(0,UTILS_DIR)
from  data_preprocess import read_txt
from  data_loader import DataLoader
from  c45_decision_tree import C45DecisionTree
#python test.py --data data/lenses_data.txt --split 1 --ckpg checkpoints/20230321_19_52_09_tree_root.npz
def parse_args():
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('--data', help='data file path')
    parser.add_argument('--split', default=450,help='split index,0-split for train,the other for test',type=int)
    parser.add_argument('--ckpg', help='max epochs to train')
    

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    data = read_txt(args,split='')
    dataloader = DataLoader(data,test=True,split_idx=[args.split])
    net = C45DecisionTree(dataloader=dataloader,load_checkpoint=args.ckpg)
    net.test()

    