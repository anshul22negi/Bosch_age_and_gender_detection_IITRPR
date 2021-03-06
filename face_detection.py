import face_recognition as fr
import torch


def get_faces(img):
    return fr.face_locations((img.permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy())
