import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
from os import listdir
from os.path import isfile, join
from torch.utils.data import DataLoader, Dataset
import os
import random

path = os.join(os.getcwd(),'dataset','mpii_human_pose_v1', 'images' )
files = sorted([join(path, f) for f in listdir(path) if isfile(join(path, f))])

random.seed(69)
random.shufle(files)

split_ratio = 0.7
train_size = split_ratio*len(files)
test_size = len(files) - train_size

transform = transforms.ToTensor()

train_imgs, test_imgs = torch.zeros([train_size, 3, 256, 256]), torch.zeros([test_size, 3, 256, 256])

for i, f in enumerate(files):
    img = Image.open(f).resize((256,256))

    if i<train_size:
        train_imgs[i, :, :] = transform(img).view((3, 256, 256))
    else:
        test_imgs[i, :, :] = transform(img).view((3, 256, 256))




class MPI(Dataset):
    def __init__(self, images, device):
        self.imgs = images

        self.imgs = self.imgs.to(device)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        return self.imgs[idx, :, :]

device = 'cpu'

trainset = MPI(train_imgs, device)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=False)          

valset = MPI(test_imgs, device)
valloader = torch.utils.data.DataLoader(valset, batch_size=32, shuffle=False)
    



