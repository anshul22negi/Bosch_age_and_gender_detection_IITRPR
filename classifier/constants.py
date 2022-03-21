import os
from os.path import dirname, abspath, isfile
from pathlib import Path
import random
import pickle as pkl

random.seed(69)

BATCH_SIZE = 128
NBINS = 10
BSETS = 10
base = Path(__file__).parents[1] / "dataset" / "utkface"

with open(Path(__file__).parents[0] / "face_locations_new.pkl","rb") as fi:
    face_map = pkl.load(fi)

bins = [[0] + sorted(random.sample(range(117), NBINS - 1)) for _ in range(BSETS)]
files = list(face_map.keys())
random.shuffle(files)

split = 0.7
train_len = int(split * len(files))
trainfiles = files[:train_len]
valfiles = files[train_len:]

device = "cuda"
