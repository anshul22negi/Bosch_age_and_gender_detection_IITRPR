import torch
from torchvision import transforms
from superres.swin_ir import SwinIR
from PIL import Image
from pathlib import Path
import argparse

model = torch.load(Path(__file__).parents[0] / "models" / "model_2.pt").cpu()

# expects (C, H, W) tensor
def upscale(img):
    with torch.no_grad():
        return model(img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("img", help="Path to image file")
    parser.add_argument("out", help="Output file path")
    args = parser.parse_args()

    transforms.ToPILImage()(upscale(transforms.ToTensor()(Image.open(args.img)))).save(
        args.out
    )
