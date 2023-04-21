import pandas as pd
import numpy as np

def read_txt(args):
    meta = np.genfromtxt(args.data,dtype='float64')

    data = meta
    features_num = data.shape[1]
    for idx in range(features_num-1):
        #最后一维是标签，不用归一化
        data[:, idx] = (data[:, idx] - np.mean(data[:, idx])) / np.std(data[:, idx])  # idx列均值方差归一化
    train_index = int(args.split)

    train_data = data[:train_index,:]
    test_data = data[train_index:,:]

    print('train_data {} * {}'.format(train_data.shape[0], train_data.shape[1]))

    print('test_data {} * {}'.format(test_data.shape[0], test_data.shape[1]))



    return train_data,test_data
    # records = pd.read_csv(args.filepath, header=None)
    # for record in records:

