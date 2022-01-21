import torch
import torch.nn as nn
import torch.optim as optim

device = "CUDA" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

def make_batch():
    input_batch = []
    target_batch = []

    for sen in sentences:
        word = sen.split()#자르고
        input = [word_dict[n] for n in word[:-1]] #인풋은 1~ n-1까지
        target = word_dict[word[-1]] # 아웃풋은 n

        input_batch.append(input)
        target_batch.append(target)

    return input_batch, target_batch #배치사이즈 만들어서 반환

# NPLM 모델 제작
class NNLM(nn.Module):
    def __init__(self): #계층(Layer)
        super(NNLM, self).__init__()
        #임베딩층
        self.C = nn.Embedding(n_class, m) #임베딩할 단어들의 갯수(룩업테이블), 임베딩할 벡터의 차원
        #레이어 층 추가
        self.H = nn.Linear(n_step * m, n_hidden, bias=False) #인풋 = 스텝수*차원 , 아웃풋 = 히든사이즈
        #파라미터 추가
        self.d = nn.Parameter(torch.ones(n_hidden)) #Bias Term
        #레이어 층 추가
        self.U = nn.Linear(n_hidden, n_class, bias=False) #인풋 = 히든사이즈, 아웃풋 = 룩업 테이블
        #레이어 층 추가
        self.W = nn.Linear(n_step * m, n_class, bias=False) #인풋 = 스텝수*차원, 아웃풋 = 룩업테이블
        #Bias 추가
        self.b = nn.Parameter(torch.ones(n_class))   #Bias Term

    def forward(self, X): #Output 반환
        X = self.C(X) #임베팅 레이어에 룩업테이블 넣음
        X = X.view(-1, n_step * m) # X를 [batch_size, n_step*m]로 형태 변경
        tanh = torch.tanh(self.d + self.H(X)) # [batch_size, n_hidden]
        output = self.b + self.W(X) + self.U(tanh) # [batch_size, n_class]
        # NPLM의 스코어 벡터 y = b + Wx + U*tanh(d+Hx)
        return output


if __name__ == '__main__':
    n_step = 2 # 스텝의 수, 논문에서는 n-1로 표현
    n_hidden = 2 # 히든사이즈 수, 논문에서의 h
    m = 2 # 임베딩사이즈, 논문에서의 m

    sentences = ["i like dog", "i love coffee", "i hate milk"]

    word_list = " ".join(sentences).split()  #join으로 한줄로 만든 다음에 split으로 자름
    word_list = list(set(word_list)) #['like', 'coffee', 'love', 'milk', 'hate', 'dog', 'i']

    word_dict = {w: i for i, w in enumerate(word_list)} #{'like': 0, 'coffee': 1, 'love': 2, 'milk': 3, 'hate': 4, 'dog': 5, 'i': 6}
    number_dict = {i: w for i, w in enumerate(word_list)} #{0: 'like', 1: 'coffee', 2: 'love', 3: 'milk', 4: 'hate', 5: 'dog', 6: 'i'}
    n_class = len(word_dict)  # 7

    model = NNLM()

    criterion = nn.CrossEntropyLoss() #다중분류
    #optim을 사용해서 가중치 갱신 방법을 구현함.
    optimizer = optim.Adam(model.parameters(), lr=0.001) #모델의 파라미터 = <generator object Module.parameters at 0x00000221A1AE2EB0>

    input_batch, target_batch = make_batch()
    input_batch = torch.LongTensor(input_batch) # type int형인 텐서로 바꾸고
    target_batch = torch.LongTensor(target_batch)

    # 학습
    for epoch in range(5000):
        #Pytorch에서는 gradients값들을 추후에 backward를 해줄때 계속 더해주기 때문"에
        #항상 backpropagation을 하기전에 gradients를 zero로 만들어주고 시작을 해야함
        optimizer.zero_grad()#변화도 버퍼를 0으로 만듬. 중요

        output = model(input_batch)

        # output : [batch_size, n_class], target_batch : [batch_size]
        #손실 함수는 (output, target)을 한 쌍(pair)의 입력으로 받아,
        #출력(output)이 정답(target)으로부터 얼마나 멀리 떨어져있는지 추정하는 값을 계산함.
        loss = criterion(output, target_batch)

        if (epoch + 1) % 1000 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

        loss.backward() #역전파하여 가중치 변화 업데이트
        optimizer.step() #업데이트 진행

    # 예측
    predict = model(input_batch).data.max(1, keepdim=True)[1]

    # 테스트
    print([sen.split()[:2] for sen in sentences], '->', [number_dict[n.item()] for n in predict.squeeze()])