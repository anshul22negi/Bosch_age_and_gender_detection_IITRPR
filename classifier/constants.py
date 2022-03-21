import os
from os.path import dirname, abspath, isfile
import random

random.seed(69)

BATCH_SIZE = 128
NBINS = 10
BSETS = 10
base = os.path.join(dirname(dirname(abspath(__file__))), "dataset", "utkface")
files = [f for f in os.listdir(base) if isfile(os.path.join(base, f))]

split = 0.7
train_len = int(split * len(files))
trainfiles = files[:train_len]
valfiles = files[train_len:]

bins = [[0] + sorted(random.sample(range(117), NBINS - 1)) for _ in range(BSETS)]
device = "cuda"