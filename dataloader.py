import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
from os import listdir
from os.path import isfile, join


class MPI(Dataset):
    def __init__(self, path: str, start: int, n_items: int, device: str):
        self.path = path
        self.start = start
        files = sorted([join(path, f) for f in listdir(path) if isfile(join(path, f))])
        self.len = (
            n_items if self.start + n_items < len(files) else len(files) - self.start
        )

        files = files[start : (start + self.len)]
        self.imgs = torch.zeros([self.len, 3, 256, 256])

        transform = transforms.ToTensor()
        for i, f in enumerate(files):
            img = Image.open(f).resize((256, 256))
            self.imgs[i, :, :] = transform(img).view((3, 256, 256))

        self.imgs = self.imgs.to(device)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.imgs[idx, :, :]
