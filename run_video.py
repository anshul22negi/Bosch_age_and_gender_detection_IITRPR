import torch
import cv2
import os
from os.path import abspath, dirname
import numpy as np
import csv
from predict_sr import upscale
from classifier.predict import predict_gender_and_age
from face_detection import get_faces
import argparse


def iou(a, b):
    inter = 0
    if a[3] <= b[1] and a[1] >= b[3] and a[0] <= b[2] and a[2] >= b[0]:
        inter = min(abs(a[3] - b[1]), abs(a[1] - b[3])) * min(
            abs(a[0] - b[2]), abs(a[2] - b[0])
        )
    else:
        return 0
    uni = (
        abs(a[1] - a[3]) * abs(a[0] - a[2])
        + abs(b[1] - b[3]) * abs(b[0] - b[2])
        - inter
    )
    return inter / uni


def process_video(video_path, do_sr, output_csv_file):
    cap = cv2.VideoCapture(video_path)

    if cap.isOpened() == False:
        print("Error opening file")

    csv_file = open(output_csv_file, "w")
    writer = csv.writer(csv_file)

    header_row = [
        "frame num",
        "person id",
        "bb_xmin",
        "bb_ymin",
        "bb_height",
        "bb_width",
        "age_min",
        "age_max",
        "age_actual",
        "gender",
    ]
    writer.writerow(header_row)

    frame_number = 0
    prev = []
    prev_fids = []
    while cap.isOpened():
        ret, frame = cap.read()
        frame_number += 1
        if ret:
            img = torch.Tensor(frame).to("cuda")
            img = img.view((1, img.shape[0], img.shape[1], img.shape[2]))
            img = img.permute(0, 3, 1, 2) / 255.0
            if do_sr:
                upscaled_img = upscale(img)
                upscaled_img = torch.squeeze(upscaled_img, dim=0)
            else:
                upscaled_img = img.squeeze(img, dim=0)
            faces = get_faces(upscaled_img)
            preds = []
            fids = []
            for face in faces:
                p = predict_gender_and_age(
                    upscaled_img[:, face[0] : face[2], face[3] : face[1]]
                )
                preds.append(p)
                done = False
                for i, old in enumerate(prev):
                    if iou(old, face) > 0.5:
                        fids.append(prev_fids[i])
                        break
                if not done:
                    fids.append(max(prev_fids + fids + [0]) + 1)

            prev = faces
            prev_fids = fids
            if len(faces) > 0:
                writer.writerows(
                    [
                        frame_number,
                        fids[i],
                        faces[i][3],
                        faces[i][0],
                        faces[i][2] - faces[i][0],
                        faces[i][1] - faces[i][3],
                        preds[i][2],
                        preds[i][3],
                        preds[i][1],
                        "M" if preds[i][0] <= 0.2 else "F",
                    ]
                    for i in range(len(faces))
                )

        else:
            break
        print("processed frame", frame_number)

    csv_file.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video", help="path to video file")
    parser.add_argument("output", help="path to csv file to write to")
    parser.add_argument("super_res", default="y", help="(y/n) whether to use super-res")
    args = parser.parse_args()
    process_video(args.video, args.super_res == "y", args.output)
