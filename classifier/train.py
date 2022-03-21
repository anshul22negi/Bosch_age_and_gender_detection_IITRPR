import torch
import torch.nn as nn
from dataloader import train_loader, test_loader
from constants import device, bins
from model import classifier_model
from losses import age_loss
from time import time
import numpy as np

output_buckets = 10
num_buckets = 10


model = classifier_model(output_buckets  =output_buckets, num_bucket_sets = num_buckets).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr =0.0001)
with open("bins.txt" ,"w") as f:
    f.write('\n'.join(' '.join(str(x) for x in y) for y in bins))
epochs = 10

age_loss_best = np.inf

for i in range(epochs):
    for img, age, gender in train_loader:

        stt = time()

        optimizer.zero_grad()

        pred_ages = model(img)
        train_loss_ages = age_loss(age, pred_ages)

        train_loss_ages.backward()

        optimizer.step()

        print(time()-stt)

    model.eval()
    with torch.no_grad():
        for img, age, gender in test_loader:
            pred_ages= model(img)
            val_loss_ages = age_loss(age, pred_ages)
    model.train()
    
    if val_loss_ages < age_loss_best:
        torch.save(model, '../models/agegender.pt')
        age_loss_best = val_loss_ages
    print(
        f"epoch: {i:3} training age loss: {train_loss_ages:10.8f} ,  validation age loss: {val_loss_ages:10.8f} "
    )

            




