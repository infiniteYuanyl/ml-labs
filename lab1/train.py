import argparse
from linear_regression import LinearRegression
from  data_preprocess import read_txt

#python train.py --data data/housing_data.txt --split 450 --epochs 1000 
def parse_args():
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('--data', help='data file path')
    parser.add_argument('--split', default=450,help='split index,0-split for train,the other for test')
    parser.add_argument('--update', help='update type')
    parser.add_argument('--epochs',default=100, help='max epochs to train')
    parser.add_argument('--ckpg', help='max epochs to train')
    parser.add_argument('--show', help='show results')
    


    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    train_data,test_data = read_txt(args)  
    net = LinearRegression(train_data, test_data,load_checkpoint=args.ckpg)
    net.train(args.epochs,args.show is not None)  

    