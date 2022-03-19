import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from os.path import dirname, abspath



import torch
import torch.nn as nn
import torch.nn.functional as F


from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms


from sklearn.model_selection import train_test_split



BATCH_SIZE = 32
data_file = os.path.join(dirname(dirname(abspath(__file__))), 'dataset', 'UTK', 'age_gender.csv')

#Reading the csv file 
df = pd.read_csv('../input/age-gender-and-ethnicity-face-data-csv/age_gender.csv')


class get_data(Dataset):
    def __init__(self, df):
        self.df = df
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            
        ])
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self,i):
        age = df['age'][i]
        gender = df['gender'][i]
        
        im = df['pixels'][i]
        im = np.reshape(im, (256,256, 3))
        im = self.transform(im)/255.0        
        age = torch.tensor(age)
        gender = torch.tensor(gender)
        
        return im, age, gender
    

train, test = train_test_split(df, test_size=0.3, random_state=69) 
train_loader = DataLoader(get_data(train), batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(get_data(test), batch_size=BATCH_SIZE, shuffle=False)

