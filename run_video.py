import torch
import cv2
import os
from os.path import abspath, dirname
import numpy as np
import csv
from predict_sr import upscale
from face_detection import get_faces
video_path = os.path.join(dirname(abspath(__file__)),'dataset', 'videoplayback.mp4')



cap = cv2.VideoCapture(video_path)

if (cap.isOpened() == False):
    print("Error opening file")

output_csv_file = os.path.join(dirname(abspath(__file__)), 'output.csv')
csv_file = open(output_csv_file, 'w')
writer = csv.writer(csv_file)

header_row = ['frame num', 'person id', 'bb_xmin', 'bb_ymin', 'bb_height', 'bb_width', 'age_min', 'age_max', 'age_actual', 'gender']
writer.writerow(header_row)

frame_number = 0
while(cap.isOpened()):
    ret, frame = cap.read()
    frame_number+=1
    if ret:
        img = torch.Tensor(frame)
        img = img.view((1, img.shape[0], img.shape[1], img.shape[2]))
        img = img.permute(0, 3, 1, 2)/255.0
        upscaled_img = upscale(img)
        
        upscaled_img = torch.squeeze(upscaled_img, dim=0)
        faces = get_faces(upscaled_img)
        print(faces)
        cv2
    else:
        break

cap.release()
cv2.destroyAllWindows()

