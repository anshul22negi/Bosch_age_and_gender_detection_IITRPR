import torch
import cv2
import os
from os.path import abspath, dirname
import numpy as np

video_path = os.path.join(dirname(abspath(__file__)),'dataset', 'videoplayback.mp4')



cap = cv2.VideoCapture(video_path)

if (cap.isOpened() == False):
    print("Error opening file")

while(cap.isOpened()):
    ret, frame = cap.read()

    if ret:
        print(ret)
        print(type(frame))
        cv2.imshow('Frame',frame)
        if cv2.waitKey(200) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()

