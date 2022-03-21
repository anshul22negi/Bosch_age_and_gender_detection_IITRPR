import torch
import torch.nn as nn
from dataloader import train_loader, test_loader
from constants import device, bins
from model import gender_classifier
from losses import age_loss
from time import time
import numpy as np
from pathlib import Path

output_buckets = 10
num_buckets = 10
load_model = False
load_path = Path(__file__).parents[0] / ".." / "models" / "agegender.pt"

if load_model:
    model = torch.load(load_path).to(device)
else:
    model = gender_classifier().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
with open("bins.txt", "w") as f:
    f.write("\n".join(" ".join(str(x) for x in y) for y in bins))
epochs = 10

loss_best = np.inf

for i in range(epochs):
    for img, age, gender in train_loader:

        stt = time()

        optimizer.zero_grad()

        pred = model(img)
        bce_loss = nn.BCELoss()
        train_loss = bce_loss(pred.squeeze(1), gender)

        train_loss.backward()

        optimizer.step()

        print(time() - stt)

    model.eval()
    with torch.no_grad():
        for img, age, gender in test_loader:
            pred = model(img)
            bce_loss = nn.BCELoss()
            val_loss = bce_loss(pred.squeeze(1), gender)
    model.train()

    if val_loss < loss_best:
        torch.save(model, "../models/gender.pt")
        loss_best = val_loss
    print(
        f"epoch: {i:3} training age loss: {train_loss:10.8f} ,  validation age loss: {val_loss:10.8f} "
    )
