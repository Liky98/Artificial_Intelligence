import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable #자동미분

torch.manual_seed(1)
context_size = 2  # {w_i-2 ... w_i ... w_i+2}
embedding_dim = 10

raw_text = """
Chang Choi is currently an Assistant Professor in the Department of Computer Engineering at Gachon University, Seongnam, Korea, Since 2020. 
He received B.S., M.S. and Ph.D. degrees in Computer Engineering from Chosun University in 2005, 2007, and 2012, respectively. 
he was a research professor at the same university. 
He was awarded the academic awards from the graduate school of Chosun University in 2012.
""".split()


def make_context_vector(context, word_to_idx): #텐서로 변환
    idxs = [word_to_idx[w] for w in context]
    return torch.tensor(idxs, dtype=torch.long)

vocab = set(raw_text) #리스트형식으로 만듬
vocab_size = len(vocab) #47개 있음.

#딕셔너리 형태로 만들기
word_to_idx = {word: i for i, word in enumerate(vocab)}
idx_to_word = {i: word for i, word in enumerate(vocab)}

data = []

for i in range(2, len(raw_text) - 2):
    context = [raw_text[i - 2], raw_text[i - 1],
               raw_text[i + 1], raw_text[i + 2]]
    target = raw_text[i]

    data.append((context, target))    #data에다가 저장.
    #ex> data = (['Chang', 'Choi', 'currently', 'an'], 'is') ...

class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CBOW, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim) #임베딩차원 설정
        self.proj = nn.Linear(embedding_dim, 128)
        self.output = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        out = self.embeddings(inputs) #단어 사이즈넣으면 [,,,,,,,] 형식으로 넣음
        out = F.relu(self.proj(out))
        out = self.output(out)
        out = F.log_softmax(out)
        return out

model = CBOW(vocab_size, embedding_dim)
optimizer = optim.SGD(model.parameters(), lr=0.001)

losses = []
loss_function = nn.NLLLoss()
loss_function1 = nn.CrossEntropyLoss()

for epoch in range(100):
    total_loss = 0
    for context, target in data:
        model.zero_grad()

        input = make_context_vector(context, word_to_idx)  # torch형식으로 만들기 > tensor[n,n,n,n]

        output = model(input)
        #loss = nn.CrossEntropyLoss(output, Variable(torch.tensor([word_to_idx[target]])))
        #loss = loss_function1(output, target)
        #loss = loss_function1(output, Variable(torch.tensor([word_to_idx[target]])))

        loss = loss_function1(output, torch.tensor(target, dtype=torch.long))

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    losses.append(total_loss)

print("*************************************************************************")

context = ['Chang', 'Choi', 'currently', 'an']
context_vector = make_context_vector(context, word_to_idx)
a = model(context_vector).data.numpy()
print('Raw text: {}\n'.format(' '.join(raw_text)))
print('Test Context: {}\n'.format(context))
max_idx = np.argmax(a)
print(context_vector)
print(max_idx)
print(len(idx_to_word))
print('Prediction: {}'.format(idx_to_word[max_idx]))
