import torch
import torch.nn as nn
from dataloader import train_loader, test_loader
from model import classifier_model
from losses import age_loss
from time import time
import numpy as np

output_buckets = 10
num_buckets = 10


model = classifier_model(output_buckets  =output_buckets, num_bucket_sets = num_buckets)
optimizer = torch.optim.Adam(model.parameters(), lr =0.0001)

epochs = 10

age_loss_best = np.inf

for i in range(epochs):
    for img, age, gender in train_loader:

        stt = time()

        optimizer.zero_grad()

        pred_ages, pred_gender = model(img)
        pred_gender = torch.sigmoid(pred_gender)
        pred_gender = torch.squeeze(pred_gender, 1)
        bce_loss = nn.BCELoss()
        train_loss_gender = bce_loss(pred_gender, gender)
        train_loss_ages = age_loss(age, pred_ages)

        train_loss_gender.backward()
        train_loss_ages.backward()

        optimizer.step()

        print(time()-stt)

    with torch.no_grad():
        for img, age, gender in test_loader:
            pred_ages, pred_gender = model(img)
            pred_gender = torch.sigmoid(pred_gender)

            val_loss_gender = nn.BCELoss(pred_gender, gender)
            val_loss_ages = age_loss(age, pred_ages)

    if val_loss_ages < age_loss_best:
        torch.save(model, 'models/model.pt')
        age_loss_best = val_loss_ages
    print(
        f"epoch: {i:3} training age loss: {train_loss_ages:10.8f} ,  validation age loss: {val_loss_ages:10.8f} "
    )
    print(
        f"epoch: {i:3} training gender loss: {train_loss_gender:10.8f} ,  validation gender loss: {val_loss_gender:10.8f} "
    )

            




