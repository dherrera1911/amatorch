import numpy as np
import torch

def load_disparity_data():
    # Load data from csv files
    s = torch.tensor(np.loadtxt('../data/dspStim.csv', delimiter=','))
    s = s.transpose(0,1)
    # Change s dtype to Double
    s = s.float()
    ctgInd = np.loadtxt('../data/dspCtg.csv', delimiter=',')
    # Change ctgInd to integer tensor and make 0-indexed
    ctgInd = torch.tensor(ctgInd, dtype=torch.int64) - 1
    ctgVal = torch.tensor(np.loadtxt('../data/dspVal.csv', delimiter=','))
    return s, ctgInd, ctgVal


