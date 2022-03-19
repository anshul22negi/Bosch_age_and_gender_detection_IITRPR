import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from os.path import dirname, abspath, isfile


import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms

import random


BATCH_SIZE = 32
NBINS = 10
BSETS = 10
base = os.path.join(dirname(dirname(abspath(__file__))), "dataset", "utkface")
files = [f for f in os.listdir(base) if isfile(join(base, f))]

random.seed(69)
split = 0.7
train_len = int(split * len(files))
trainfiles = files[:train_len]
valfiles = files[train_len:]

bins = [[0] + sorted(random.sample(range(117), NBINS - 1)) for _ in range(NSETS)]
device = "cpu"


class get_data(Dataset):
    def __init__(self, f):
        self.files = [join(base, x) for x in f]
        self.ages = [int(x.split("_")[0]) for x in f]
        for i in range(len(f)):
            vals = torch.zeros(NBINs, dtype=torch.float32)
            for j in range(NSETS):
                for k in range(NBINS):
                    if self.ages[i] >= bins[j][k]:
                        vals[j] = k
            self.ages[i] = vals.to(device)

        self.genders = [
            F.one_hot(torch.tensor(int(x.split("_")[1])), num_classes=2).to(device)
            for x in f
        ]
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.f)

    def __getitem__(self, i):
        img = Image.open(self.files[i])

        return self.transform(img).to(device), self.ages[i], self.genders[i]


train_loader = DataLoader(get_data(trainfiles), batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(get_data(valfiles), batch_size=BATCH_SIZE, shuffle=False)
