import os
from keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
from keras.applications import inception_v3
import datetime
import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
import tensorflow
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras import Input
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras import regularizers
from tensorflow.keras import optimizers
import random
import tensorflow as tf
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import OneHotEncoder
# from sklearn.model_selection import train_test_splits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from collections import deque
import pandas as pd

path = "C:/Users/LeeKihoon/PycharmProjects/gachonProject/기계학습프로그래밍/img/Fruit/"
print("싱싱한 귤 Train 데이터 량 : ", len(os.listdir(path+"train/fresh")))
print("썪은 귤 Train 데이터 량 : ", len(os.listdir(path+"train/rotten")))
print("싱싱한 귤 validation 데이터 량 : ", len(os.listdir(path+"validation/fresh")))
print("썪은 귤 validation 데이터 량 : ", len(os.listdir(path+"validation/rotten")))
print("싱싱한 귤 Test 데이터 량 : ", len(os.listdir(path+"test/fresh")))
print("썪은 귤 Test 데이터 량 : ", len(os.listdir(path+"test/rotten")))

#%%

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    path+"train", target_size=(150,150), batch_size=1, class_mode='binary'
)
validation_generator = test_datagen.flow_from_directory(
    path+"validation", target_size=(150,150), batch_size=1, class_mode='binary'
)
for data_batch, label_batch in train_generator:

    print("데이터 Shape : ", data_batch.shape)
    print("라벨 shape : ", label_batch.shape)
    break

for v_data_batchm, v_label_batch in validation_generator :
    break
#%%

#%%

 #%%
""" Setting """
# 전부 랜덤 시드 주기
random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)

# 강화학습에 필요한 값들 설정
DISCOUNT = 0.9  # 누적보상할인계수
REPLAY_MEMORY_SIZE = 2000  # 리플레이메모리 최대사이즈
MIN_REPLAY_MEMORY_SIZE = 512  # 256으로 설정가능. 최소사이즈
MIN_BATCH_SIZE = 128  # 최소 배치사이즈
EPISODES = 100  # 에피소드
epsilon = 0.9  # 앱실론 설정 (무작위탐험)
epsilon_decay = 0.99  # 0.8로도 설정가능. 앱실론에 곱해서 점점 줄여나감
MIN_EPSILON = 0.0001
BATCH_SIZE = 32  # 배치사이즈
ALPHA = 1
ALPHA_DECAY = 0.9  # 1 0.9 # 0.9999 #0.9975
ALPHA_MIN = 0.0001
ssc = StandardScaler()  # 평균을 제거하고 데이터를 단위 분산으로 조정
ssc = MinMaxScaler()  # 모든 feature 값이 0~1사이에 있도록 데이터를 재조정

pretrainingModel_inception_v3 = inception_v3.InceptionV3(weights='imagenet',
                  include_top=False,
                  input_shape=(150, 150, 3))
pretrainingModel_inception_v3.trainable = True

set_trainable = False
for layer in pretrainingModel_inception_v3.layers:
    if layer.name == 'conv2d_95':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False


class DQN():
    def __init__(self, input_size, output_size, data, labels):
        self.model = self.create_model(input_size, output_size)#기본모델
        self.target_model = self.create_model(input_size, output_size)#타겟모델
        self.obs = data #환경은 데이터
        self.labels = labels #라벨
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        pass

    def create_model(self, input_size, output_size):
        model = models.Sequential()
        model.add(pretrainingModel_inception_v3)
        model.add(layers.Flatten())
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dense(2, activation=None))

        # #  input_tensor = pretrainingModel_inception_v3()
      #   x = layers.add(pretrainingModel_inception_v3)
      #   x = layers.Flatten()(x)
      #   x = layers.Dropout(0.1)(x)
      #   x = layers.Dense(256, activation='relu')(x)
      #   x = layers.Dense(64, activation='relu')(x)
      #   x = layers.Dense(16, activation='relu')(x)
      #   x = layers.Dense(output_size, activation=None)(x)
      #   model = Model(inputs=input_tensor, outputs=x)

        model.compile(loss='mse', optimizer=optimizers.RMSprop(), metrics=['mae'])
        return model

    def play(self, obs, action):
        r = np.where((self.obs == obs).all(axis=1)) #열에서 환경이 모두 같냐?
        if len(r[0]) == 0:
            # something really wrong observation not in the game
            print("........... SOMETHING WRONG ................")
            return -100
        if self.labels[r[0][0]] == action:
            return 100
        else:
            return -100

    def update_replay(self, obs, action, label, reward):
        self.replay_memory.append(obs, action, label, reward)

    def get_qs(self, obs):
        return self.model.predict(obs, batch_size=1)

    def train(self):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        minibatch = random.sample(self.replay_memory, MIN_BATCH_SIZE)
        y = np.array([np.array(xi, dtype=np.dtype(object)) for xi in minibatch])
        z = np.array([np.array(xi) for xi in y[:, 0]])

        current_qa_list = dqn.model.predict(z, batch_size=BATCH_SIZE)
        future_qa_list = dqn.target_model.predict(z, batch_size=BATCH_SIZE)

        max_future_qa = np.argmax(future_qa_list, axis=1)
        alpha = ALPHA
        # qa = np.zeros(shape=(minibatch.shape[0],3))

        for indx in range(y.shape[0]):
            # alpha1=1 gives 97 accuracy
            #current_qa_list[indx, y[indx, 1]] = (alpha1 * y[indx,3]) + ((1-alpha1)*max_future_qa[indx])
            # y[indx,1] is the action
            # y[indx,3] is the reward
            # z is the states / observation in numpy format
            current_qa_list[indx, y[indx, 1]] = (alpha * y[indx,3]) + ((1-alpha)*max_future_qa[indx])
            #current_qa_list[indx, y[indx, 1]] = current_qa_list[indx, y[indx, 1]] + (alpha * (y[indx, 3]) - current_qa_list[indx, y[indx, 1]])
            pass

        dqn.model.fit(z, current_qa_list, epochs=16, batch_size=BATCH_SIZE, verbose=0)
        return

#%%
dqn = DQN(data_batch.shape[0], 2, train_generator, label_batch)
c = 0
total_rewards = 0

while True:
    indx = np.random.randint(0, data_batch.shape[0])
    act = data_batch[indx]
    act = dqn.get_qs(data_batch[indx].reshape(1, -1))  #모델로 액션결정하기

    if np.random.rand() > epsilon: #랜덤으로 앱실론보다 크면
        act = np.argmax(act) #둘중 최대값 선택해서 진행
    else:
        act = np.random.randint(0, high=2) #무작위 탐험 걸리면 랜덤 액션
    r = dqn.play(data_batch[indx], action=act)

    dqn.update_replay(data_batch[indx], act, data_batch[indx], r)
    dqn.train()
    total_rewards = total_rewards + r

    if c%100 == 0 :
        print(c,"번째 에피소드 100까지의 총합 리워드 : ",total_rewards)
        total_rewards = 0
    if total_rewards > 9500 :
        break
    c = c + 1
    if ALPHA > ALPHA_MIN:
        ALPHA *= ALPHA_DECAY
    else:
        ALPHA = ALPHA_MIN

    if epsilon > MIN_EPSILON:
        epsilon *= epsilon_decay
    else:
        epsilon = MIN_EPSILON

print("총 ", c, "번의 에피소드가 있었습니다.")
#%%
results = np.argmax(dqn.model.predict(data_batch, batch_size=BATCH_SIZE), axis=1)
a = accuracy_score(data_batch, results)
print(" train accuracy ", a)
tn, fp, fn, tp = confusion_matrix(data_batch, results).ravel()
print(" train tn ,", tn, " fp ", fp, " fn ", fn, " tp ", tp)

results = np.argmax(dqn.model.predict(v_data_batchm, batch_size=BATCH_SIZE), axis=1)
a = accuracy_score(v_label_batch, results)
print(" Test accuracy ", a)
tn, fp, fn, tp = confusion_matrix(v_label_batch, results).ravel()
print(" train tn ,", tn, " fp ", fp, " fn ", fn, " tp ", tp)

a,b =dqn.model.evaluate(validation_generator)
print("검증 데이터 모델 정확도 : ", b)


