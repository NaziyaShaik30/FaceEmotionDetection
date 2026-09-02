import torch.nn as nn
import torch.nn.functional as F

class DeepANN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(DeepANN, self).__init__()
        self.fc1 = nn.Linear(input_size * 50, 256)  # Multiply input size with time steps
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.view(x.shape[0], -1)  # Flatten input (batch_size, 50*40)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class SimpleANN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SimpleANN, self).__init__()
        self.fc1 = nn.Linear(input_size * 50, 128)  # Multiply input size with time steps
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.view(x.shape[0], -1)  # Flatten input (batch_size, 50*40)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
