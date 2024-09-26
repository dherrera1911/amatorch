import os
import torch
import numpy as np

def load_data(data_dir):

    # Load data from csv files
    stimuli = torch.tensor(
      np.loadtxt(
        os.path.join(data_dir, 'stimuli.csv'),
        delimiter=',')
    )
 
    stimuli = stimuli.transpose(0,1)

    # Change s dtype to Double
    s = s.float()
    ctgInd = np.loadtxt('../data/dspCtg.csv', delimiter=',')
    # Change ctgInd to integer tensor and make 0-indexed
    ctgInd = torch.tensor(ctgInd, dtype=torch.int64) - 1
    ctgVal = torch.tensor(np.loadtxt('../data/dspVal.csv', delimiter=','))
    return s, ctgInd, ctgVal


