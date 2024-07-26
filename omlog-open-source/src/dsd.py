from .MMD import mmd
import numpy as np
import torch


def dsd_judge(batch_history, batch_current):
    loss = mmd(torch.tensor(np.array(batch_history['eventids'])), torch.tensor(np.array(batch_current['eventids'])))
    return loss