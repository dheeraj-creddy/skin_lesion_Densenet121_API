from torchvision import models
import torch.nn as nn

def get_model(num_classes=7):
    model = models.densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    #replaces the original DenseNet classifier output layer with
    # a new Linear layer whose output features equal the number of classes
    # in your specific dataset (e.g., 7 for HAM10000).
    return model
