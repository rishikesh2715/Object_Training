import torch.nn as nn
import torchvision.models as models

def get_model(num_classes, use_y_channel=False):
    model = models.resnet18(weights=None)
    
    if use_y_channel:
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model