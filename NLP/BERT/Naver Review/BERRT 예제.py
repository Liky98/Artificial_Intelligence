import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_transformers import BertTokenizer, BertForSequenceClassification, BertConfig
from torch.optim import Adam
import torch.nn.functional as F

train_df = pd.read_csv('./ratings_train.txt', sep='\t')
test_df = pd.read_csv('./ratings_test.txt', sep='\t')

train_df.info # 150,000 Rows, 3 Columns

train_df = train_df.dropna()
test_df = test_df.dropna()

train_df = train_df.sample(frac=0.4, random_state=999) #40%만 가지고오기
test_df = test_df.sample(frac=0.4, random_state=999) #40%만 가져오기

train_df.info #59,998 rows, 3 Columns
test_df.info #19,999 rows, 3 Columns

class load_dataset(Dataset) :
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        text = self.df.iloc[item, 1]
        label = self.df.iloc[item,2]
        return text, label

train_dataset = load_dataset(train_df)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

device = torch.device("cuda")
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased')
model.to(device)

optimizer = Adam(model.parameters(), lr=1e-6)

""" 모델 학습 """
model.train()

total_corrct = 0
total_len = 0
total_loss = 0
count = 0
for epoch in range(1) :
    for text, label in train_loader:
        optimizer.zero_grad()

        encoding_list = [tokenizer.encode(x, add_special_tokens=True) for x in text] #한 문장에서 단어씩
                                                                                #<CLS>, <SEP> 등의 special token을 추가
        padding_list = [x + [0]*(512-len(x)) for x in encoding_list]

        sample = torch.tensor(padding_list)
        sample = sample.to(device)
        label = torch.tensor(label)
        label = label.to(device)

        outputs = model(sample, labels = label)
        loss, logits = outputs

        predict = torch.argmax(F.softmax(logits), dim=1)
        corrct = predict.eq(label)

        total_corrct += corrct.sum().item()
        total_len += len(label)
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

        if count % 1000 ==0 :
            print(f'Epoch : {epoch+1}, Iteration : {count}')
            print(f'Train Loss : {total_loss/1000}')
            print(f'Accuracy : {total_corrct/total_len}\n')

        count +=1

""" 모델 평가 및 결과"""

model.eval()

test_dataset = load_dataset(test_df)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

total_corrct = 0
total_len = 0

for text, label in test_loader :
    encoding_list = [tokenizer.encode(x, add_special_tokens=True) for x in text]
    padding_list = [x + [0]*(512-len(x)) for x in encoding_list]

    sample = torch.tensor(padding_list)
    sample = sample.to(device)
    label = torch.tensor(label)
    label = label.to(device)

    outputs = model(sample, labels=label)
    loss, logits = outputs

    predict = torch.argmax(F.softmax(logits), dim=1)
    corrct = predict.eq(label)

    total_corrct += corrct.sum().item()
    total_len += len(label)

print(f'Test Accuracy : {total_corrct/total_len}')

#%%
torch.save(model.state_dic(), "bert_naver_review.model")
#%%
model.load_state_dict(torch.load("bert_naver_review.model"))
model.eval()