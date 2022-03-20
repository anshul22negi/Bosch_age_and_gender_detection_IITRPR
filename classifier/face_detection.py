import cv2 as cv
from pathlib import Path
import torch
import torchvision.transforms as transforms

detector = cv.CascadeClassifier(str(Path(__file__).parents[0] / "haarcascade_frontalface_default.xml"))


def get_faces(img):
    if img.shape[0] == 3:
        img = transforms.Grayscale()(img)
    
    return detector.detectMultiScale(
        img
        .mul(256)
        .type(torch.uint8)
        .permute((1, 2, 0))
        .numpy(),
        1.1,
        4,
    )
