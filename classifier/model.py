import torch
import torchvision.models as models
from constants import device
import torch.nn as nn


class classifier_model(nn.Module):
    def __init__(self, output_buckets, num_bucket_sets):
        super(classifier_model, self).__init__()
        self.model_ft = models.vgg16(pretrained=True).to(device)
        self.n_buckets = output_buckets
        self.n_sets = num_bucket_sets

        self.pre_age_classifier_layer = nn.Sequential(
            nn.Linear(1000, 256, bias=True),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, 128, bias=True),
        ).to(device)

        self.bin_layers = [
            nn.Sequential(nn.Linear(128, output_buckets, bias=False), nn.Softmax(dim=1)).to(device)
            for _ in range(num_bucket_sets)
        ]

    def forward(self, x):
        with torch.no_grad():
            x = self.model_ft(x)

        age = self.pre_age_classifier_layer(x)
        y = torch.zeros((x.shape[0], self.n_sets, self.n_buckets), dtype=x.dtype).to(device)
        for i in range(self.n_sets):
            y[:, i, :] = self.bin_layers[i](age)
        return y

class gender_classifier(nn.Module):
    def __init__(self):
        super(gender_classifier, self).__init__()
        self.pre_gender_feature_layer = nn.Sequential(
            nn.Conv2d(3, 64, 3),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 3, stride =3),
            
            nn.Conv2d(64, 128, 3),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 3, stride =3),
            
            nn.Conv2d(128, 256, 3),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 3, stride =3)
        ).to(device)
        
        self.classifier_layer = nn.Sequential(
            nn.Linear(in_features = 256, out_features = 4096, bias = True),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(in_features = 4096, out_features = 1000),
            nn.Linear(1000, 256, bias=True),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 128, bias=True),
            nn.Linear(128, 1),
            nn.Sigmoid()
        ).to(device)
    
    def forward(self, x):
        x = self.pre_gender_feature_layer(x).view((x.shape[0], -1))
        gender = self.classifier_layer(x)
        return gender
