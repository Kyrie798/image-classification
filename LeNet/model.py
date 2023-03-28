import torch.nn as nn

class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.subsampling1 = nn.Sequential(nn.Conv2d(3, 16, 5),
                                          nn.ReLU(),
                                          nn.MaxPool2d(2, 2))
        
        self.subsampling2 = nn.Sequential(nn.Conv2d(16, 32, 5),
                                          nn.ReLU(),
                                          nn.MaxPool2d(2, 2))

        self.fc1 = nn.Linear(32 * 5 * 5, 120)
        self.relu3 = nn.ReLU()

        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()

        self.fc3 = nn.Linear(84, 10)
    
    def forward(self, x):
        # x:[batch, 3, 32, 32]
        x = self.subsampling1(x)

        x = self.subsampling2(x)

        x = x.view(-1, 32 * 5 * 5)
        x = self.fc1(x)
        x = self.relu3(x)

        x = self.fc2(x)
        x = self.relu4(x)

        x = self.fc3(x)
        return x