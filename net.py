import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, out_dim = 3):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 40 * 30, 128)  # Assuming input image size of 320x240
        self.fc2 = nn.Linear(128, out_dim)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 128 * 40 * 30)  # Flattening before fully connected layer
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return self.tanh( x )