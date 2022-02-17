"""인코더(Encoder) 아키텍처
주어진 소스 문장을 문맥 벡터(context vector)로 인코딩합니다.

LSTM은 hidden state과 cell state을 반환합니다.

하이퍼 파라미터(hyperparameter)

input_dim: 하나의 단어에 대한 원핫 인코딩 차원
embed_dim: 임베딩(embedding) 차원
hidden_dim: 히든 상태(hidden state) 차원
n_layers: RNN 레이어의 개수
dropout_ratio: 드롭아웃(dropout) 비율
"""

import torch.nn as nn

# 인코더(Encoder) 아키텍처 정의
class Encoder(nn.Module):
    def __init__(self, input_dim, embed_dim, hidden_dim, n_layers, dropout_ratio):
        super().__init__()

        # 임베딩(embedding)은 원-핫 인코딩(one-hot encoding)을 특정 차원의 임베딩으로 매핑하는 레이어
        self.embedding = nn.Embedding(input_dim, embed_dim)

        # LSTM 레이어
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.rnn = nn.LSTM(embed_dim, hidden_dim, n_layers, dropout=dropout_ratio)

        # 드롭아웃(dropout)
        self.dropout = nn.Dropout(dropout_ratio)

    # 인코더는 소스 문장을 입력으로 받아 문맥 벡터(context vector)를 반환
    def forward(self, src):
        # src: [단어 개수, 배치 크기]: 각 단어의 인덱스(index) 정보
        embedded = self.dropout(self.embedding(src))
        # embedded: [단어 개수, 배치 크기, 임베딩 차원]

        outputs, (hidden, cell) = self.rnn(embedded)
        # outputs: [단어 개수, 배치 크기, 히든 차원]: 현재 단어의 출력 정보
        # hidden: [레이어 개수, 배치 크기, 히든 차원]: 현재까지의 모든 단어의 정보
        # cell: [레이어 개수, 배치 크기, 히든 차원]: 현재까지의 모든 단어의 정보

        # 문맥 벡터(context vector) 반환
        return hidden, cell