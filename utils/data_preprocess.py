import pandas as pd
import numpy as np

def read_txt(args,split=''):
    meta = np.genfromtxt(args.data,dtype='float64',delimiter=split)
    return meta