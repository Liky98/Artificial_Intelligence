import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
import torch
import torch.nn as nn
import torch.optim as optim
from keras.models import Sequential
from keras.layers import Dense, Activation
import torch
import torch.nn as nn
import torch.optim as optim

device = "CUDA" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

train_data = 'you need to know how to code'

def processing(text) :
    # 중복을 제거한 단어들의 집합인 단어 집합 생성.
    word_set = set(text.split())
    # 단어 집합의 각 단어에 고유한 정수 맵핑.
    vocab = {tkn: i+2 for i, tkn in enumerate(word_set)}
    vocab['<unk>'] = 0
    vocab['<pad>'] = 1
    return vocab

def make_batch():
    input_batch = []
    target_batch = []

    for sen in sentences:
        word = sen.split()
        input = [word_dict[n] for n in word[:-1]] #인풋은 1~ n-1까지
        target = word_dict[word[-1]] # 아웃풋은 n

        input_batch.append(input)
        target_batch.append(target)

    return input_batch, target_batch

class NNLM(nn.Module):
    def __init__(self):
        super(NNLM, self).__init__()
        self.Embedding = nn.Embedding(n_class, m) #embedding
        self.Hidden01 = nn.Linear(n_step * m, n_hidden, bias=False)    #히든레이어 가중치
        self.Ones = nn.Parameter(torch.ones(n_hidden))         #Ones
        self.Hidden02 = nn.Linear(n_hidden, n_class, bias=False)
        self.Hidden03 = nn.Linear(n_step * m, n_class, bias=False) #가중치
        self.Bias = nn.Parameter(torch.ones(n_class))          #Bias Term

    def forward(self, X):
        X = self.Embedding(X) # X : [batch_size, n_step, m]
        X = X.view(-1, n_step * m) # [batch_size, n_step * m]
        tanh = torch.tanh(self.d + self.Hidden01(X)) # [batch_size, n_hidden]
        output = self.b + self.Hidden03(X) + self.Hidden02(tanh) # [batch_size, n_class]
        # NPLM의 스코어 벡터 y = b + Wx + U*tanh(d+Hx)
        return output

if __name__ == '__main__' :
    n_step = 2
    m = 2
    n_hidden = 2

    excel = pd.read_csv("Emotion.csv")
    text = excel["Text"] #21459,
    sentenses = list(set(text))# 21405,

    n_class = len(sentenses)  # number of Vocabulary

    model = NNLM()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    input_batch, target_batch = make_batch()
    input_batch = torch.LongTensor(input_batch)
    target_batch = torch.LongTensor(target_batch)
