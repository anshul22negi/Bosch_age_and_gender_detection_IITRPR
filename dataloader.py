import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
from os import listdir
from os.path import isfile, join
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import random

path = Path(__file__).parents[0] / "dataset" / "mpii_human_pose_v1" / "images"
files = sorted([join(path, f) for f in listdir(path) if isfile(join(path, f))])

random.seed(69)
random.shuffle(files)

split_ratio = 0.7
train_size = int(split_ratio * len(files))
test_size = len(files) - train_size

transform = transforms.ToTensor()

train_imgs, test_imgs = files[:train_size], files[train_size:]


class MPI(Dataset):
    def __init__(self, images, device):
        self.imgs = images
        self.device = device

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = Image.open(self.imgs[idx])
        return transform(img.resize((128, 128))).view((3, 128, 128)).to(
            self.device
        ), transform(img.resize((256, 256))).view((3, 256, 256)).to(self.device)


device = "cpu"

trainset = MPI(train_imgs, device)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=8, shuffle=False)

valset = MPI(test_imgs, device)
valloader = torch.utils.data.DataLoader(valset, batch_size=8, shuffle=False)
