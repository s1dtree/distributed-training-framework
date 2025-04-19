import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    """
    Simple CNN for classifying CIFAR-10 images

    Architecture:
    - 3 convolutional layers w/ RELU activation & max pooling
    - 2 fully connected layers
    - Output layer w/ 10 classes
    """
    def __init__(self):
        super(SimpleCNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)

        # Max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        self.fc1 = nn.Linear(in_features=128*4*4, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=10)

        # Dropout for regularization
        self.dropout = nn.Dropout(p=0.25)

    def forward(self, x):
        """
        Forward pass through the network

        Args:
            x: Input tensor of shape [batch_size, 3, 32, 32]

        Returns:
            Output tensor of shape [batch_size, 10]
        """

        # First conv block: conv -> relu -> pool
        x = self.pool(F.relu(self.conv1(x))) # Output: [batch_size, 3, 16, 16]

        # Second conv block: conv -> relu -> pool
        x = self.pool(F.relu(self.conv2(x))) # Output: [batch_size, 64, 8, 8]

        # Second conv block: conv -> relu -> pool
        x = self.pool(F.relu(self.conv3(x))) # Output: [batch_size, 128, 4, 4]

        # Flatten output from conv layers
        x = x.view(-1, 128*4*4) # Output: [batch_size, 128*4*4]

        # First fully connected layer w/ dropout
        x = self.dropout(F.relu(self.fc1(x))) # Output: [batch_size, 512]

        # Second fully connected layer w/ dropout
        x = self.fc2(x) # Output: [batch_size, 10]

        return x