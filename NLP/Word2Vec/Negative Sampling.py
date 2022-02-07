## using pytorch
import torch
import torch.nn as nn
import random
example_sentence = """
Chang Choi is currently an Assistant Professor in the Department of Computer Engineering at Gachon University, Seongnam, Korea, Since 2020. 
He received B.S., M.S. and Ph.D. degrees in Computer Engineering from Chosun University in 2005, 2007, and 2012, respectively. 
he was a research professor at the same university. 
He was awarded the academic awards from the graduate school of Chosun University in 2012.
""".split()

#(1) 입력받은 문장을 단어로 쪼개고, 중복을 제거해줍니다.
vocab = set(example_sentence)
vocab_size = len(example_sentence)

#(2) 단어 : 인덱스, 인덱스 : 단어를 가지는 딕셔너리를 선언해 줍니다.
word_to_index = {word:index for index, word in enumerate(vocab)}
index_to_word = {index:word for index, word in enumerate(vocab)}

#(3) 학습을 위한 데이터를 생성해 줍니다.
# convert context to index vector
def make_context_vector(context, word_to_ix):
  idxs = word_to_ix[context] # chang을 넣으면 몇번째 인덱스인지
  return torch.tensor(idxs, dtype=torch.long)

# make dataset function

#랜덤으로 뽑기
def make_data(sentence):
  data = []
  for i in range(2, len(example_sentence) - 2):
    context = example_sentence[i]
    target = [random.choice(example_sentence),
              random.choice(example_sentence),
              random.choice(example_sentence),
              random.choice(example_sentence),
              ]
    data.append((context, target))
  return data

data = make_data(example_sentence)

#(4) Skip-Gram 모델을 정의해 줍니다.
class SKIP_GRAM(nn.Module):
  def __init__(self, vocab_size, embedding_dim, context_size):
    super(SKIP_GRAM, self).__init__()
    self.context_size = context_size
    self.embeddings = nn.Embedding(vocab_size, embedding_dim)

    self.layer1 = nn.Linear(embedding_dim, 64)
    self.activation1 = nn.ReLU()

    self.layer2 = nn.Linear(64, vocab_size * context_size)
    self.activation2 = nn.LogSoftmax(dim = -1)

  def forward(self, inputs):
    embeded_vector = self.embeddings(inputs)
    output = self.activation1(self.layer1(embeded_vector))
    output = self.activation2(self.layer2(output))
    return output.view(self.context_size,vocab_size)

#(5) 모델을 선언해주고, loss function, optimizer등을 선언해줍니다.
EMBEDDING_DIM = 128
EPOCHS = 200
CONTEXT_SIZE = 4

model = SKIP_GRAM(vocab_size, EMBEDDING_DIM, CONTEXT_SIZE)
loss_function = nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

#(6) 학습을 진행합니다.
for epoch in range(EPOCHS):
    total_loss = 0
    for context, target in data:
        context_vector = make_context_vector(context, word_to_index)
        log_probs = model(context_vector)
        total_loss += loss_function(log_probs, torch.tensor([word_to_index[t] for t in target]))
    print('epoch = ',epoch, ', loss = ',total_loss)
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

#(7) test하고 싶은 문장을 뽑고, test를 진행합니다.
test_data = 'Computer'
test_vector = make_context_vector(test_data, word_to_index)
result = model(test_vector)
print(f"Input : {test_data}")
print('Prediction : ', [index_to_word[torch.argmax(r).item()] for r in result])