import torch
from torchvision import transforms
from superres.swin_ir import SwinIR
from PIL import Image
from pathlib import Path
import argparse

model = torch.load(
    Path(__file__).parents[0] / "models" / "model_2.pt",
    map_location=torch.device("cpu"),
).cuda()

# expects (C, H, W) tensor
def upscale(img):
    with torch.no_grad():
        return model(img)
