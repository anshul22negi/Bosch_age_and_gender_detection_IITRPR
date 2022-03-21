import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from classifier.model import classifier_model, gender_classifier
from classifier.constants import bins
from pathlib import Path
import argparse
import math

device = "cuda"


def preprocess_image(img):
    return transforms.functional.resize(img.to(device), (64, 64)).view((1, 3, 64, 64))


age_model_path = Path(__file__).parents[0] / ".." / "models" / "agegender_2.pt"
age_model = classifier_model(10, 10)
age_model.load_state_dict(torch.load(age_model_path))
age_model.eval()
gender_model_path = Path(__file__).parents[0] / ".." / "models" / "gender.pt"
gender_model = gender_classifier()
gender_model.load_state_dict(torch.load(gender_model_path))
gender_model.eval()


def predict_gender_and_age(img):
    image = preprocess_image(img)
    age_prediction = 0.0

    with torch.no_grad():
        pred_ages = age_model(image)
        age_low, age_high = 0, 116
        pred_ages = torch.squeeze(pred_ages, 0)
        for i in range(pred_ages.size()[0]):
            for j in range(pred_ages.size()[1]):
                if j < pred_ages.size()[1] - 1:
                    age_prediction += (
                        (bins[i][j] + bins[i][j + 1]) * pred_ages[i][j] / 2
                    )
                else:
                    age_prediction += (bins[i][j] + 116) * pred_ages[i][j] / 2
        age_prediction /= 10
        for i in range(pred_ages.size()[0]):
            max_confidence = 0
            temp_low, temp_high = 0, 0
            for j in range(pred_ages.size()[1]):
                if j < pred_ages.size()[1] - 1:
                    if (
                        age_prediction >= bins[i][j]
                        and age_prediction <= bins[i][j + 1]
                        and max_confidence < pred_ages[i][j]
                    ):
                        max_confidence, temp_low, temp_high = (
                            pred_ages[i][j],
                            bins[i][j],
                            bins[i][j + 1],
                        )
                else:
                    if (
                        age_prediction >= bins[i][j]
                        and age_prediction <= 116
                        and max_confidence < pred_ages[i][j]
                    ):
                        max_confidence, temp_low, temp_high = (
                            pred_ages[i][j],
                            bins[i][j],
                            116,
                        )
            age_low += temp_low
            age_high += temp_high

        pred_gender = gender_model(image)[0].item()
        age_low /= 10
        age_high /= 10
    return (
        pred_gender,
        math.floor(age_prediction.item()),
        math.floor(age_low),
        math.floor(age_high),
    )
