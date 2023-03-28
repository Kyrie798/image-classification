import torch
import torch.nn as nn

class AlexNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 48, 11, stride=4, padding=2),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(3, 2),
                                      nn.Conv2d(48, 128, 5, padding=2),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(3, 2),
                                      nn.Conv2d(128, 192, 3, padding=1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(192, 192, 3, padding=1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(192, 128, 3, padding=1),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(3, 2),
                                      )
        self.classifier = nn.Sequential(nn.Dropout(p=0.5),
                                        nn.Linear(128 * 6 * 6, 2048),
                                        nn.ReLU(inplace=True),
                                        nn.Dropout(p=0.5),
                                        nn.Linear(2048, 2048),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(2048, num_classes),
                                        )
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x