import argparse
from linear_regression import LinearRegression
from  data_preprocess import read_txt

#python test.py --data data/housing_data.txt --split 450 --ckpg checkpoints/20230319_00_16_19_paramsW.npz
def parse_args():
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('--data', help='data file path')
    parser.add_argument('--split', default=450,help='split index,0-split for train,the other for test')
    parser.add_argument('--ckpg', help='max epochs to train')
    

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    train_data,test_data = read_txt(args)  
    net = LinearRegression(train_data, test_data,load_checkpoint=args.ckpg)
    net.test()

    