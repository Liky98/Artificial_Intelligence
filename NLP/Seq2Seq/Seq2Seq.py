"""
Seq2Seq 아키텍처
앞서 정의한 인코더(encoder)와 디코더(decoder)를 가지고 있는 하나의 아키텍처입니다.
인코더(encoder): 주어진 소스 문장을 문맥 벡터(context vector)로 인코딩합니다.
디코더(decoder): 주어진 문맥 벡터(context vector)를 타겟 문장으로 디코딩합니다.
단, 디코더는 한 단어씩 넣어서 한 번씩 결과를 구합니다.
Teacher forcing: 디코더의 예측(prediction)을 다음 입력으로 사용하지 않고, 실제 목표 출력(ground-truth)을 다음 입력으로 사용하는 기법
"""
import torch
import torch.nn as nn
import random
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    # 학습할 때는 완전한 형태의 소스 문장, 타겟 문장, teacher_forcing_ratio를 넣기
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src: [단어 개수, 배치 크기]
        # trg: [단어 개수, 배치 크기]
        # 먼저 인코더를 거쳐 문맥 벡터(context vector)를 추출
        hidden, cell = self.encoder(src)

        # 디코더(decoder)의 최종 결과를 담을 텐서 객체 만들기
        trg_len = trg.shape[0]  # 단어 개수
        batch_size = trg.shape[1]  # 배치 크기
        trg_vocab_size = self.decoder.output_dim  # 출력 차원
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        # 첫 번째 입력은 항상 <sos> 토큰
        input = trg[0, :]

        # 타겟 단어의 개수만큼 반복하여 디코더에 포워딩(forwarding)
        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell)

            outputs[t] = output  # FC를 거쳐서 나온 현재의 출력 단어 정보
            top1 = output.argmax(1)  # 가장 확률이 높은 단어의 인덱스 추출

            # teacher_forcing_ratio: 학습할 때 실제 목표 출력(ground-truth)을 사용하는 비율
            teacher_force = random.random() < teacher_forcing_ratio
            input = trg[t] if teacher_force else top1  # 현재의 출력 결과를 다음 입력에서 넣기

        return outputs