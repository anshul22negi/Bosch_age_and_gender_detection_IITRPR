import numpy as np  # linear algebra
import os
from os.path import dirname, abspath, isfile
from pathlib import Path


import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
from constants import *

import pickle as pkl
import random


class get_data(Dataset):
    def __init__(self, f):
        self.transform = transforms.ToTensor()
        self.resize = transforms.Resize((64, 64))

        files = [Path(base) / x for x in f]
        self.faces = [face_map[x] for x in f]

        self.images = [
            self.resize(
                transforms.functional.crop(
                    self.transform(Image.open(files[i])),
                    self.faces[i][0],
                    self.faces[i][3],
                    self.faces[i][2] - self.faces[i][0],
                    self.faces[i][1] - self.faces[i][3],
                )
            ).to(device)
            for i in range(len(files))
        ]

        self.ages =torch.tensor([int(x.name.split("_")[0]) for x in files]).to(device)
        self.agevs = [0 for x in files]
        for i in range(len(files)):
            vals = torch.zeros(NBINS, dtype=torch.int32)
            for j in range(BSETS):
                for k in range(NBINS):
                    if self.ages[i] >= bins[j][k]:
                        vals[j] = k
            self.agevs[i] = vals.to(device)

        self.genders = [
            torch.tensor(int(x.split("_")[1]), dtype=torch.float32).to(device)
            for x in f
        ]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        return self.images[i], self.ages[i], self.agevs[i], self.genders[i]


train_loader = DataLoader(get_data(trainfiles), batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(get_data(valfiles), batch_size=BATCH_SIZE, shuffle=False)
