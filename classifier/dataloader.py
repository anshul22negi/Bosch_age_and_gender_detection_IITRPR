import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
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

from face_detection import get_faces

import pickle as pkl
import random


BATCH_SIZE = 128
NBINS = 10
BSETS = 10
base = os.path.join(dirname(dirname(abspath(__file__))), "dataset", "utkface")
files = [f for f in os.listdir(base) if isfile(os.path.join(base, f))]

random.seed(69)
split = 0.7
train_len = int(split * len(files))
trainfiles = files[:train_len]
valfiles = files[train_len:]

bins = [[0] + sorted(random.sample(range(117), NBINS - 1)) for _ in range(BSETS)]
device = "cpu"


class get_data(Dataset):
    def __init__(self, f):
        self.transform = transforms.ToTensor()
        with open(Path(__file__).parents[0] / "face_locations.pkl", "rb") as fi:
            face_map = pkl.load(fi)

        self.files = [os.path.join(base, x) for x in f if x in face_map]
        self.faces = [face_map[x] for x in f if x in face_map]

        self.ages = [int(x.split("_")[0]) for x in f]
        for i in range(len(f)):
            vals = torch.zeros(NBINS, dtype=torch.int32)
            for j in range(BSETS):
                for k in range(NBINS):
                    if self.ages[i] >= bins[j][k]:
                        vals[j] = k
            self.ages[i] = vals.to(device)

        self.genders = [
            torch.tensor(int(x.split("_")[1]), dtype=torch.float32).to(device)
            for x in f
        ]
        self.resize = transforms.Resize((64,64))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        img = self.transform(Image.open(self.files[i]))
        img = self.resize(transforms.functional.crop(img, self.faces[i][0], self.faces[i][1],self.faces[i][2],self.faces[i][3]))

        return img.to(device), self.ages[i], self.genders[i]


train_loader = DataLoader(get_data(trainfiles), batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(get_data(valfiles), batch_size=BATCH_SIZE, shuffle=False)
