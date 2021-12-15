#%%
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

#%% 전처리하기
train = pd.read_csv('C:/Users/LeeKihoon/PycharmProjects/gachonProject/Q-learning/Titanic/train.csv')
test = pd.read_csv('C:/Users/LeeKihoon/PycharmProjects/gachonProject/Q-learning/Titanic/test.csv')

labels = train['Survived'] #라벨은 생존
train.head()
#%% 이름은 필요없으니 지움
train01 = train
train01 = train01.drop(["Survived"], axis=1)
train01 = train01.drop(["Name"], axis=1)
train01 = train01.drop(["PassengerId"], axis=1)
train01 = train01.drop(["Ticket"], axis=1)
train01 = train01.drop(["Cabin"],axis=1)
train01.head()
#%% 결측치 확인해보기
train01.isnull().sum()
#%% 결측치 대체
#train01['Cabin'] = train01['Cabin'].fillna('N')
train01['Age'] = train01['Age'].fillna(0)
train01['Embarked'] = train01['Embarked'].fillna('N')
#%% 각 컬럼 속성 확인
train01.info()
#%% 타입 변경하기
train02 = train01
train02 = pd.get_dummies(data=train02, columns= ['Embarked'], prefix='Embarked')
train02 = pd.get_dummies(data=train02, columns= ['Sex'], prefix='Sex')
#%%

train02['Age'] = list(map(int,train02['Age']))
train02['Fare'] = list(map(int,train02['Fare']))

train02.info()

#%%
""" Setting """
    # 전부 랜덤 시드 주기
random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)
    # 강화학습에 필요한 값들 설정
DISCOUNT = 0.9 #누적보상할인계수
REPLAY_MEMORY_SIZE = 2000 #리플레이메모리 최대사이즈
MIN_REPLAY_MEMORY_SIZE = 512  # 256으로 설정가능. 최소사이즈
MIN_BATCH_SIZE = 128 #최소 배치사이즈
EPISODES = 100 #에피소드
epsilon = 0.9  # 앱실론 설정 (무작위탐험)
epsilon_decay = 0.99  # 0.8로도 설정가능. 앱실론에 곱해서 점점 줄여나감
MIN_EPSILON = 0.0001
BATCH_SIZE = 32 #배치사이즈
ALPHA = 1
ALPHA_DECAY = 1  # 1 0.9 # 0.9999 #0.9975
ALPHA_MIN = 0.0001
ssc = StandardScaler()   #평균을 제거하고 데이터를 단위 분산으로 조정
ssc = MinMaxScaler()     #모든 feature 값이 0~1사이에 있도록 데이터를 재조정
digitizer = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='kmeans') #10개의 구간으로 kmeans하게 나눔

#%%
z = np.where(labels == 1, 100, -100)
train03 = train02
train03.shape
#%%
#oe = OneHotEncoder(handle_unknown='ignore', sparse=False)
#train_labels= oe.fit_transform(labels.values.reshape(-1,1)) ## 여기가 에러임
###train_data = digitizer.fit_transform(train03)
#%%
train_data = ssc.fit_transform(train03)
train_data.shape
#%%

#%%
class DQN():
    def __init__(self, input_size, output_size, data, labels):
        self.model = self.create_model(input_size, output_size)
        self.target_model = self.create_model(input_size, output_size)
        self.obs = data
        self.labels = labels
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        pass

    def create_model(self, input_size, output_size):

        input_tensor = Input(shape=(input_size,))
        x = layers.Dense(32, activation='relu')(input_tensor)
        x = layers.Dropout(0.1)(x)
        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dense(16, activation='relu')(x)
        x = layers.Dropout(0.1)(x)
        x = layers.Dense(16, activation='relu')(x)
        x = layers.Dropout(0.1)(x)
        x = layers.Dense(8, activation='relu')(x)
        x = layers.Dropout(0.1)(x)
        output_tensor = layers.Dense(output_size, activation=None)(x)
        model = Model(inputs=input_tensor, outputs=output_tensor)
        # print (model.summary())
        ad = optimizers.Adam(learning_rate=0.001)  # 'rmsprop'
        model.compile(optimizer=ad,
                      loss='mse', metrics=['mae'])
        return model

    def play(self, obs, action):
        r = np.where((self.obs == obs).all(axis=1))
        if len(r[0]) == 0:
            # something really wrong observation not in the game
            print("........... SOMETHING WRONG ................")
            return -100
        if self.labels[r[0][0]] == action:
            return 100
        else:
            return -100

    def update_replay(self, obs, action, label, reward):
        self.replay_memory.append(((obs), action, label, reward))

    def get_qs(self, obs):
        return self.model.predict(obs, batch_size=1)

    def train(self):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return
        minibatch = random.sample(self.replay_memory, MIN_BATCH_SIZE)
        # conevrt minibatch in list type to y numpy array. This is needed for access
        # also keras and TF except in numpy format
        y = np.array([np.array(xi, dtype=np.dtype(object)) for xi in minibatch])
        # extract features as a numpy array. As needed by Keras.
        # shape is (batch_size, feature_size)
        z = np.array([np.array(xi) for xi in y[:, 0]])
        current_qa_list = dqn.model.predict(z, batch_size=BATCH_SIZE)
        future_qa_list = dqn.target_model.predict(z, batch_size=BATCH_SIZE)
        max_future_qa = np.argmax(future_qa_list, axis=1)
        alpha = ALPHA
        # qa = np.zeros(shape=(minibatch.shape[0],3))
        # Here only update the argmax because in theory we select max action
        # hence only the Q value of argmax get affected.
        # 이론상 최대 행동을 선택하기 때문에 argmax만 업데이트합니다.
        # 따라서 argmax의 Q값만 영향을 받습니다.
        for indx in range(y.shape[0]):
            # alpha1=1 gives 97 accuracy
            # current_qa_list[indx, y[indx, 1]] = (alpha1 * y[indx,3]) + ((1-alpha1)*max_future_qa[indx])
            # y[indx,1] is the action
            # y[indx,3] is the reward
            # z is the states / observation in numpy format
            # current_qa_list[indx, y[indx, 1]] = (alpha * y[indx,3]) + ((1-alpha)*max_future_qa[indx])
            current_qa_list[indx, y[indx, 1]] = current_qa_list[indx, y[indx, 1]] + (
                        alpha * (y[indx, 3]) - current_qa_list[indx, y[indx, 1]])
            pass
        dqn.model.fit(z, current_qa_list, epochs=16, batch_size=BATCH_SIZE, verbose=0)
        return

    # dqn = DQN(train_data.shape[1], 2, train_data, labels)


#%%
print(train_data.shape)
labels.shape
#%%
train_data, test_data, train_labels, test_labels = train_test_split(train_data, labels, test_size=0.3,random_state=True)

#%%

#%%
dqn = DQN(train_data.shape[1], 2, train_data, train_labels)
c = 0
total_rewards = 0
#%%
while c < 1200:
    indx = np.random.randint(0, train_data.shape[0])

    act = train_labels[indx]
    act = dqn.get_qs(train_data[indx].reshape(1, -1))
    print("act : ", act)
    if np.random.rand() > epsilon:
        act = np.argmax(act)
    else:
        act = np.random.randint(0, high=2)
    r = dqn.play(train_data[indx], action=act)
    print("act2 : ", act)
    dqn.update_replay(train_data[indx], act, train_labels[indx], r)
    dqn.train()
    total_rewards = total_rewards + r
    if c%100 == 0 :
        print(c,"번째 에피소드 reward : ", total_rewards)
    c = c + 1
    if ALPHA > ALPHA_MIN:
        ALPHA *= ALPHA_DECAY
    else:
        ALPHA = ALPHA_MIN

    if epsilon > MIN_EPSILON:
        epsilon *= epsilon_decay
    else:
        epsilon = MIN_EPSILON

#%%
results = np.argmax(dqn.model.predict(train_data, batch_size=BATCH_SIZE), axis=1)
a = accuracy_score(train_labels, results)
print(" train accuracy ", a)
tn, fp, fn, tp = confusion_matrix(train_labels, results).ravel()
print(" train tn ,", tn, " fp ", fp, " fn ", fn, " tp ", tp)
results = np.argmax(dqn.model.predict(test_data, batch_size=BATCH_SIZE), axis=1)
a = accuracy_score(test_labels, results)
print(" Test accuracy ", a)
tn, fp, fn, tp = confusion_matrix(test_labels, results).ravel()
print(" train tn ,", tn, " fp ", fp, " fn ", fn, " tp ", tp)
