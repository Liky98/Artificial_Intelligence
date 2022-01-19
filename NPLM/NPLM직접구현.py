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

"""
class NNLM(nn.Module):
    def __init__(self):
        super(NNLM, self).__init__()
        self.C = nn.Embedding(n_class, m)
        self.H = nn.Linear(n_step * m, n_hidden, bias=False)    #히든레이어 가중치
        self.d = nn.Parameter(torch.ones(n_hidden))         #Bias Term
        self.U = nn.Linear(n_hidden, n_class, bias=False)
        self.W = nn.Linear(n_step * m, n_class, bias=False) #가중치
        self.b = nn.Parameter(torch.ones(n_class))          #Bias Term

    def forward(self, X):
        X = self.C(X) # X : [batch_size, n_step, m]
        X = X.view(-1, n_step * m) # [batch_size, n_step * m]
        tanh = torch.tanh(self.d + self.H(X)) # [batch_size, n_hidden]
        output = self.b + self.W(X) + self.U(tanh) # [batch_size, n_class]
        # NPLM의 스코어 벡터 y = b + Wx + U*tanh(d+Hx)
        return output
"""
### Tensorflow로
C = tf.Variable(tf.random_normal_initializer([1]), name='C')
model = Sequential()
model.add(Dense(units=64), input_dim=28*28, activarion='relu')
model.compile(loss='categorical_crossentropy',optimizer='sgd', metrics=['accuracy'])


###

def load_excel(path) :
    return pd.read_csv(path)

def load_data(path):
    #데이터 불러오는 함수
    with open(path, 'r', encoding='utf-8') as f:
        data = f.read()
    return data

def score_vector():
    # 스코어 벡터 제작
    #y = b + Wx + U*tanh(d+Hx)
    print()

def n_gram(text):
    #n-gram 구현
    words = text.split()

    for i in range(len(words) -1 ) :
        print(words[i], words[i+1])

    n_gram_zip = list(zip(words, words[1:]))
    return n_gram_zip

def lookup_table() :
    print()

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

excel = load_excel("Emotion.csv")
text = excel["Text"] #21459,
voca = list(set(text))# 21405,

tokenizer = Tokenizer()

for sentence in voca :
    tokenizer.fit_on_texts([sentence])
    print(f"단어는 {tokenizer.word_index}")
    print(f"인코딩은 {tokenizer.texts_to_sequences([sentence])[0]}")
    print(to_categorical(tokenizer.texts_to_sequences([sentence])[0]))

