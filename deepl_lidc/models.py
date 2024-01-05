""" 
Module with neural network models.
"""

import torch
from torch import nn

softmax = nn.Softmax(dim=1)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(84*84, 3528),
            nn.ReLU(),
            nn.Linear(3528, 3528),
            nn.ReLU(),
            nn.Linear(3528, 5),
        )
        
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        pred_prob = softmax(logits)
        return pred_prob
        
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=7, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(10, 20, kernel_size=3, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(20, 40, kernel_size=3, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2560, 1280),
            nn.ReLU(),
            nn.Linear(1280, 1280),
            nn.ReLU(),
            nn.Linear(1280, 1280),
            nn.ReLU(),
            nn.Linear(1280, 160),
            nn.ReLU(),
            nn.Linear(160, 5)
        )

    def forward(self, x):
        x = self.features(x.unsqueeze(1)) #unsqueeze add channel dimension
        x = self.classifier(x)
        pred = softmax(x)
        return pred