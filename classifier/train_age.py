import torch
import torch.nn as nn
from constants import device, bins
print(bins)
from dataloader import train_loader, test_loader
from model import classifier_model
from losses import age_loss
from time import time
import numpy as np
from pathlib import Path
from dataloader import bins

def return_age_pred(pred_ages, bins ):
    age_low,age_high = 0, 116
    max_confidence = 0
    age_prediction=torch.zeros(pred_ages.size()[0]).to(device)
    for i in range(pred_ages.size()[1]):
        for j in range(pred_ages.size()[1]):
            if j < pred_ages.size()[1]-1:
                age_prediction += (bins[i][j] + bins[i][j+1])*pred_ages[:,i,j]/2
            else:
                age_prediction += (bins[i][j] + 116)*pred_ages[:,i,j]/2
        
    '''
    for i in range(pred_ages.size()[0]):
        for j in range(pred_ages.size()[1]):
            if j < pred_ages.size()[1]-1:
                if age_prediction >= bins[i][j] and age_prediction <= bins[i][j+1] and max_confidence<pred_ages[:,i,j]:
                        max_confidence, age_low, age_high = pred_ages[:,i,j], bins[i][j], bins[i][j+1]
            else:
                if age_prediction >= bins[i][j] and age_prediction <= 116 and max_confidence<pred_ages[:,i,j]:
                    max_confidence, age_low, age_high = pred_ages[:,i,j], bins[i][j], 116
    '''
                        


                    
    return age_prediction/10

output_buckets = 10
num_buckets = 10
load_model = True
load_path = Path(__file__).parents[0] / ".." / "models" / "agegender.pt"

if load_model:
    model = torch.load(load_path).to(device)
else:
    model = classifier_model(
        output_buckets=output_buckets, num_bucket_sets=num_buckets
    ).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
with open("bins.txt", "w") as f:
    f.write("\n".join(" ".join(str(x) for x in y) for y in bins))
epochs = 10

age_loss_best = np.inf

for i in range(epochs):
    msel = 0
    mse_loss = nn.MSELoss()
    total_train_mse, total_val_mse = 0,0
    for img, act_age, age, gender in train_loader:

        stt = time()

        optimizer.zero_grad()

        pred_ages = model(img)
        train_loss_ages = age_loss(age, pred_ages)
        predicted_age = return_age_pred(pred_ages, bins)
        train_mse_loss = mse_loss(act_age, predicted_age)
        total_train_mse += train_mse_loss.item()
        train_loss_ages.backward()

        optimizer.step()

        print(time() - stt)

    model.eval()
    with torch.no_grad():
        for img, act_age, age, gender in test_loader:
            pred_ages = model(img)
            val_loss_ages = age_loss(age, pred_ages)
            predicted_age = return_age_pred(pred_ages, bins)
            val_mse_loss = mse_loss(act_age, predicted_age)
            total_val_mse += val_mse_loss.item()
    model.train()

    if val_loss_ages < age_loss_best:
        torch.save(model, "../models/agegender.pt")
        age_loss_best = val_loss_ages
    print(
        f"epoch: {i:3} training age loss: {train_loss_ages:10.8f} ,  validation age loss: {val_loss_ages:10.8f} "
    )
    print(
        f"epoch: {i:3} training mse loss: {total_train_mse:10.8f} ,  validation mse loss: {total_val_mse:10.8f} "
    )
