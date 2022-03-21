import torch
import cv2
import os
from os.path import abspath, dirname
import numpy as np
import csv

video_path = os.path.join(dirname(abspath(__file__)),'dataset', 'videoplayback.mp4')



cap = cv2.VideoCapture(video_path)

if (cap.isOpened() == False):
    print("Error opening file")

output_csv_file = ""
csv_file = open(output_csv_file, 'wb')
writer = csv.writer(csv_file)

header_row = ['frame num', 'person id', 'bb_xmin', 'bb_ymin', 'bb_height', 'bb_width', 'age_min', 'age_max', 'age_actual', 'gender']
writer.writerow(header_row)

frame_number = 0
while(cap.isOpened()):
    ret, frame = cap.read()
    frame_number+=1
    if ret:
        img = torch.Tensor(frame)

        cv2.imshow('Frame',frame)
        
    else:
        break

cap.release()
cv2.destroyAllWindows()

