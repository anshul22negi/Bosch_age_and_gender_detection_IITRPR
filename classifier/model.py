import torch
import torchvision.models as models
import torch.nn as nn


class classifier_model(nn.Module):
    def __init__(self, output_buckets, num_bucket_sets):
        super(classifier_model, self).__init__()
        self.model_ft = models.vgg16(pretrained=True)
        self.n_buckets = output_buckets
        self.n_sets = num_bucket_sets

        self.pre_age_classifier_layer = nn.Sequential(
            nn.Linear(1000, 256, bias=True),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, 128, bias=True),
        )

        self.bin_layers = [
            nn.Sequential(nn.Linear(128, output_buckets, bias=False), nn.LogSoftmax(dim=1))
            for _ in range(num_bucket_sets)
        ]

        self.gender_classifier_layer = nn.Sequential(
            nn.Linear(1000, 256, bias=True),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 128, bias=True),
            nn.Linear(128, 2)
            
        )

    def forward(self, x):
        with torch.no_grad():
            x = self.model_ft(x)

        age = self.pre_age_classifier_layer(x)
        y = torch.zeros((x.shape[0], self.n_sets, self.n_buckets), dtype=x.dtype)
        for i in range(self.n_sets):
            y[:, i, :] = self.bin_layers[i](age)
        
        gender = self.gender_classifier_layer(x)
         
        return y, gender


if __name__ == "__main__":
    model = classifier_model(5)
    x = torch.randn((32, 3, 256, 256))
    x, y = model(x)
    print(x.shape)
