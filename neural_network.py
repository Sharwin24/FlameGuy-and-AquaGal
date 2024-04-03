import torch
import torch.nn as nn

# initialzing convolutional neural network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 4, stride = 2)
        self.conv2 = nn.Conv2d(16, 32, 4, stride = 2)
        self.conv3 = nn.Conv2d(32, 64, 4, stride = 2)
        self.pool = nn.MaxPool2d(2, 2)
        #self.fc0 = nn.Linear(3400 * 64, 3400)
        self.fc1 = nn.Linear(2240, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 6)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = torch.flatten(x, 0)
        #x = self.relu(self.fc0(x))
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x