# -*- coding: utf-8 -*-
"""Sentence Transformer를 이용한 챗봇 구현.ipynb
Automatically generated by Colaboratory.
Original file is located at
    https://colab.research.google.com/drive/11_7XmcSqs3KcXrlijl4BHzrPQ8xMocIt
"""


import urllib.request
import pandas as pd
import numpy as np
from numpy import dot
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer

urllib.request.urlretrieve("https://raw.githubusercontent.com/songys/Chatbot_data/master/ChatbotData.csv", filename="ChatBotData.csv")
train_data = pd.read_csv('ChatBotData.csv')
train_data.head()

model = SentenceTransformer('sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens')

train_data['embedding'] = train_data.apply(lambda row: model.encode(row.Q), axis = 1)

def cos_sim(A, B):
  return dot(A, B)/(norm(A)*norm(B))

def return_similar_answer(input):
    embedding = model.encode(input)
    train_data['score'] = train_data.apply(lambda x: cos_sim(x['embedding'], embedding), axis=1)

#%%
print(return_similar_answer('안녕하세요'))#