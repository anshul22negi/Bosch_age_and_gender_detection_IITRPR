import torch
import torch.nn as nn
import numpy as np


def age_loss(age, y_pred):
    loss = 0
    for i in range(y_pred.shape[0]):
        for j in range(age.shape[1]):
            loss += y_pred[i, j, age[i, j]]

    return -1*loss          


    

     
