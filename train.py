import torch
import torch.nn as nn
from dataloader import trainloader, valloader
from swin_ir import SwinIR
from utils.psnr import calculate_psnr
from time import time

input_height, input_width = 128, 128  # Actual dimensions of image from DataLoader
upscale = 2
window_size = 8

# Height and Width of each patch
height = (input_height // upscale // window_size + 1) * window_size
width = (input_width // upscale // window_size + 1) * window_size


model = SwinIR(
    upscale=upscale,
    img_size=(input_height, input_width),
    patch_size=(height, width),
    window_size=window_size,
    img_range=1.0,
    depths=[4, 4,4, 4],
    embed_dim=60,
    num_heads=[4,4,4,4],
    mlp_ratio=2,
    upsampler="pixelshuffledirect",
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

epochs = 10

for i in range(epochs):

    for seq in trainloader:
        stt = time()
        x, y = seq

        optimizer.zero_grad()
        y_pred = model(x)
        single_loss = calculate_psnr(y_pred, y)

        single_loss.backward()
        optimizer.step()

        train_loss = single_loss.item()
        print(stt - time())

    with torch.no_grad():
        for seq in valloader:
            x, y = seq
            y_pred = model(x)
            single_loss = calculate_psnr(y_pred, y)

            val_loss = single_loss.item()

    print(
        f"epoch: {i:3} training loss: {train_loss:10.8f} ,  validation loss: {val_loss:10.8f} "
    )
