import torch
import torch.nn as nn
import numpy as np


def age_loss(age, y_pred):
    loss = 0
    log_prob = torch.log10(nn.softmax(y_pred, dim = 1))
    for i,log_probs in enumerate(log_prob):
        for j in range(len(age[i])):
            if age[j] == 1:
                loss+= log_probs[j]
    return -1*loss


    

     
