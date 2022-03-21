import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from dataloader import bins
from pathlib import Path
device = 'cpu'

def preprocess_image(file_name):
    img = Image.open(file_name).resize((512,512))
    transform = transforms.ToTensor()
    
    return transform(img).to(device)



model_path = Path(__file__) / ".." / "temp" / "agegender.pt"
model  = torch.load(model_path)

def predict_gender_and_age(image_path):
    image  = preprocess_image(image_path)
    age_prediction = 0.0



    with torch.no_grad():
        pred_ages, pred_gender = model()
        age_low,age_high = 0, 116
        max_confidence = 0
        pred_ages = torch.squeeze(pred_ages,0)
        for i in range(pred_ages.size()[0]):
            for j in range(pred_ages.size()[1]):
                if j < pred_ages.size()[1]-1:
                    age_prediction += (bins[i][j] + bins[i][j+1])*pred_ages[i][j]/2
                else:
                    age_prediction += (bins[i][j] + 116)*pred_ages[i][j]/2
        

        for i in range(pred_ages.size()[0]):
            for j in range(pred_ages.size()[1]):
                if j < pred_ages.size()[1]-1:
                    if age_prediction >= bins[i][j] and age_prediction <= bins[i][j+1] and max_confidence<pred_ages[i][j]:
                        max_confidence, age_low, age_high = pred_ages[i][j], bins[i][j], bins[i][j+1]
                else:
                    if age_prediction >= bins[i][j] and age_prediction <= 116 and max_confidence<pred_ages[i][j]:
                        max_confidence, age_low, age_high = pred_ages[i][j], bins[i][j], 116

                        


                    
    return pred_gender,age_prediction, age_low, age_high


if __name__ == '__main__':
    image_path = ''
    pred_gender, age_prediction = predict_gender_and_age(image_path)
    print(f"gender: {pred_gender}  ||    age: {age_prediction}")



