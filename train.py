import torch
import torch.nn as nn
from dataloader import trainloader, valloader
from swin_ir import SwinIR
from utils.psnr import calculate_psnr

input_height, input_width = 256, 256 #Actual dimensions of image from DataLoader
upscale = 4
window_size = 8

#Height and Width of each patch
height = (input_height // upscale // window_size + 1) * window_size
width = (input_width // upscale // window_size + 1) * window_size


model =     model = SwinIR(
        upscale=2,
        img_size=(height, width),
        window_size=window_size,
        img_range=1.0,
        depths=[6, 6, 6, 6],
        embed_dim=60,
        num_heads=[6, 6, 6, 6],
        mlp_ratio=2,
        upsampler="pixelshuffledirect",
    )

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

epochs = 10

for i in range(epochs):
    
    for seq in trainloader:
        labels = seq

        optimizer.zero_grad()
        y_pred = model(seq)
        single_loss = calculate_psnr(y_pred, labels)
        
        single_loss.backward()
        optimizer.step()

        train_loss = single_loss.item()
    
    for seq in valloader:
        labels = seq
        y_pred - model(seq)
        single_loss = calculate_psnr(y_pred, labels)

        val_loss = single_loss.item()


    print(f'epoch: {i:3} training loss: {train_loss:10.8f} ,  validation loss: {val_loss:10.8f} ')