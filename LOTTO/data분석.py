import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

df = pd.read_excel('Data.xlsx')
df.head()

def incoding(num) :
    zero_list = np.zeros(45)
    for i in range(6):
        zero_list[int(num[i])-1]=1
    return zero_list

def decoding(num_list):
    numbers = []
    for i in range(num_list):
        if num_list[i] ==1.0 :
            numbers.append(i+1)

    return numbers

train_idx = (0,900)
val_idx = (901,1000)
test = (1001, len(df))

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
model = NeuralNetwork().to(device)
print(model)
