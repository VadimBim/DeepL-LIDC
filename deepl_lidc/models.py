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