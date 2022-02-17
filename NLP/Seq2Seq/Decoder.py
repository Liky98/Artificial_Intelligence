"""디코더(Decoder) 아키텍처
주어진 문맥 벡터(context vector)를 타겟 문장으로 디코딩합니다.
LSTM은 hidden state과 cell state을 반환합니다.
하이퍼 파라미터(hyperparameter)
input_dim: 하나의 단어에 대한 원핫 인코딩 차원
embed_dim: 임베딩(embedding) 차원
hidden_dim: 히든 상태(hidden state) 차원
n_layers: RNN 레이어의 개수
dropout_ratio: 드롭아웃(dropout) 비율
"""
import torch.nn as nn
# 디코더(Decoder) 아키텍처 정의
class Decoder(nn.Module):
    def __init__(self, output_dim, embed_dim, hidden_dim, n_layers, dropout_ratio):
        super().__init__()

        # 임베딩(embedding)은 원-핫 인코딩(one-hot encoding) 말고 특정 차원의 임베딩으로 매핑하는 레이어
        self.embedding = nn.Embedding(output_dim, embed_dim)

        # LSTM 레이어
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.rnn = nn.LSTM(embed_dim, hidden_dim, n_layers, dropout=dropout_ratio)

        # FC 레이어 (인코더와 구조적으로 다른 부분)
        self.output_dim = output_dim
        self.fc_out = nn.Linear(hidden_dim, output_dim)

        # 드롭아웃(dropout)
        self.dropout = nn.Dropout(dropout_ratio)

    # 디코더는 현재까지 출력된 문장에 대한 정보를 입력으로 받아 타겟 문장을 반환
    def forward(self, input, hidden, cell):
        # input: [배치 크기]: 단어의 개수는 항상 1개이도록 구현
        # hidden: [레이어 개수, 배치 크기, 히든 차원]
        # cell = context: [레이어 개수, 배치 크기, 히든 차원]
        input = input.unsqueeze(0)
        # input: [단어 개수 = 1, 배치 크기]

        embedded = self.dropout(self.embedding(input))
        # embedded: [단어 개수, 배치 크기, 임베딩 차원]

        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        # output: [단어 개수 = 1, 배치 크기, 히든 차원]: 현재 단어의 출력 정보
        # hidden: [레이어 개수, 배치 크기, 히든 차원]: 현재까지의 모든 단어의 정보
        # cell: [레이어 개수, 배치 크기, 히든 차원]: 현재까지의 모든 단어의 정보

        # 단어 개수는 어차피 1개이므로 차원 제거
        prediction = self.fc_out(output.squeeze(0))
        # prediction = [배치 크기, 출력 차원]

        # (현재 출력 단어, 현재까지의 모든 단어의 정보, 현재까지의 모든 단어의 정보)
        return prediction, hidden, cell