import torch
from torchvision import transforms
from swin_ir import SwinIR
from PIL import Image

model = torch.load("models/model_2.pt")

# expects (C, H, W) tensor
def upscale(path):
    img = transforms.ToTensor()(Image.open(path))
    with torch.no_grad():
        return model(img)
