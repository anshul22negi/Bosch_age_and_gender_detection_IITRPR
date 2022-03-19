import torch
import torchvision.models as models
import torch.nn as nn


class classifier_model(nn.Module):
    def __init__(self, output_buckets):
        super(classifier_model, self).__init__()
        self.model_ft = models.vgg16(pretrained = True)

        
        self.age_classifier_layer = nn.Sequential(
            nn.Linear(1000, 256, bias = True), 
             nn.BatchNorm1d(256),
             nn.Dropout(0.2),
             nn.Linear(256 , 128, bias = True),
             nn.Linear(128 , output_buckets)
             
        )

        self.gender_classifier_layer = nn.Sequential(
            nn.Linear(1000, 256, bias = True), 
             nn.BatchNorm1d(256),
             nn.Dropout(0.3),
             nn.Linear(256 , 128, bias = True),
             nn.Linear(128 , output_buckets)
             
        )   
        
    def forward(self, x):
        x = self.model_ft(x)
        
        age = self.age_classifier_layer(x)
        gender = self.gender_classifier_layer(x)
        
        return age, gender
    
    
if __name__ == '__main__':
    model = classifier_model(5)
    x = torch.randn((32,3, 256, 256))
    x,y = model(x)
    print(x.shape)
        
        
    


