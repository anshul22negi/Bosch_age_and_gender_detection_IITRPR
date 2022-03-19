import cv2 as cv
from pathlib import Path
import torch
import torchvision.transforms as transforms

detector = cv.CascadeClassifier("haarcascade_frontalface_default.xml")


def get_faces(img):
    return detector.detectMultiScale(
        transforms.Grayscale()(img)
        .mul(256)
        .type(torch.uint8)
        .permute((1, 2, 0))
        .numpy(),
        1.1,
        4,
    )
