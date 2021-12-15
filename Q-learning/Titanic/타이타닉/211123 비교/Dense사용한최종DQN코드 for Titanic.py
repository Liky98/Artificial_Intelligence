"""
    Train Accuracy : 92.616
    Validation Accuracy : 79.85
    Test Accuracy : 73.444
    Episod 수 : 10700번

     train      :   tn  381  fp  15  fn  31  tp  196
     validation :   tn  135  fp  18  fn  36  tp  79
                                            """
# %%
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

# %% 전처리하기
train = pd.read_csv('C:/Users/LeeKihoon/PycharmProjects/gachonProject/Q-learning/Titanic/train.csv')
test = pd.read_csv('C:/Users/LeeKihoon/PycharmProjects/gachonProject/Q-learning/Titanic/test.csv')
#train = pd.read_csv(
#    '/content/drive/Othercomputers/310호 IIP/PycharmProjects/gachonProject/Q-learning/Titanic/train.csv')
#test = pd.read_csv('/content/drive/Othercomputers/310호 IIP/PycharmProjects/gachonProject/Q-learning/Titanic/test.csv')

labels = train['Survived'].values  # labels == 생존했는지

train01 = train  # 백업

train01 = train01.drop(["Survived"], axis=1)
train01 = train01.drop(["Name"], axis=1)
train01 = train01.drop(["PassengerId"], axis=1)
train01 = train01.drop(["Ticket"], axis=1)
train01 = train01.drop(["Cabin"], axis=1)  # 전처리귀찮은거 다버려

# 결측치 대체
train01['Age'] = train01['Age'].fillna(0)
train01['Embarked'] = train01['Embarked'].fillna('S')
# 정수로 전처리
train01 = pd.get_dummies(data=train01, columns=['Embarked'], prefix='Embarked')
train01 = pd.get_dummies(data=train01, columns=['Sex'], prefix='Sex')
# 2차 백업
train02 = train01

train_data = (train02 - train02.mean(axis=0)) / train02.std(axis=0)
train_data
#%%
""" 이런식으로 변환해도 됌
from sklearn.preprocessing import LabelEncoder # 추가 추가 추가
train['Sex'] = LabelEncoder().fit_transform(train['Sex'])
test['Sex'] = LabelEncoder().fit_transform(test['Sex'])
"""


#%%
""" Setting """
    # 전부 랜덤 시드 주기
random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)

    # 강화학습에 필요한 값들 설정
DISCOUNT = 0.9 #누적보상할인계수
REPLAY_MEMORY_SIZE = 2000 #리플레이메모리 최대사이즈
MIN_REPLAY_MEMORY_SIZE = 512
MIN_BATCH_SIZE = 128
EPISODES = 100 #에피소드
epsilon = 0.9  # 앱실론 설정 (무작위탐험)
epsilon_decay = 0.99  # 0.8로도 설정가능. 앱실론에 곱해서 점점 줄여나감
MIN_EPSILON = 0.0001
BATCH_SIZE = 32 #배치사이즈
ALPHA = 1
ALPHA_DECAY = 0.9  # 1 0.9 # 0.9999 #0.9975
ALPHA_MIN = 0.0001

#%%
class DQN():
    def __init__(self, input_size, output_size, data, labels):
        self.model = self.create_model(input_size, output_size)#기본모델
        self.target_model = self.create_model(input_size, output_size)#타겟모델
        self.obs = data #환경은 데이터
        self.labels = labels #라벨
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        pass

    def create_model(self, input_size, output_size):
        input_tensor = Input(shape=(input_size,))
        x = layers.Dense(64, activation='relu')(input_tensor)
        x = layers.Dropout(0.1)(x)
        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dropout(0.1)(x)
        x = layers.Dense(16, activation='relu')(x)
        x = layers.Dropout(0.1)(x)
        x = layers.Dense(8, activation='relu')(x)
        x = layers.Dropout(0.1)(x)
        x = layers.Dense(4, activation='relu')(x)
        x = layers.Dropout(0.1)(x)
        output_tensor = layers.Dense(output_size, activation=None)(x) #아웃풋은 2개의 확률이나옴.  Q스코어.
        model = Model(inputs=input_tensor, outputs=output_tensor)

        ad = optimizers.Adam(learning_rate=0.001)  # or rmsprop
        model.compile(optimizer=ad,
                      loss='mse', metrics=['mae'])
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
        self.replay_memory.append(((obs), action, label, reward))

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
        #qa = np.zeros(shape=(minibatch.shape[0],3))

        for indx in range(y.shape[0]):
            # alpha1=1 gives 97 accuracy
            # current_qa_list[indx, y[indx, 1]] = (alpha1 * y[indx,3]) + ((1-alpha1)*max_future_qa[indx])
            # y[indx,1] is the action
            # y[indx,3] is the reward
            # z is the states / observation in numpy format
            current_qa_list[indx, y[indx, 1]] = (alpha * y[indx,3]) + ((1-alpha)*max_future_qa[indx])
            current_qa_list[indx, y[indx, 1]] = current_qa_list[indx, y[indx, 1]] + (alpha * (y[indx, 3]) - current_qa_list[indx, y[indx, 1]])
            pass

        dqn.model.fit(z, current_qa_list, epochs=16, batch_size=BATCH_SIZE, verbose=0)
        return

#%%
train_data, test_data, train_labels, test_labels = train_test_split(train_data, labels, test_size=0.3,random_state=True)
dqn = DQN(train_data.shape[1], 2, train_data, train_labels)
c = 0
total_rewards = 0
#%% test
train_data01 = pd.DataFrame.to_numpy(train_data)
test_data01 =  pd.DataFrame.to_numpy(test_data)

#%%
total_rewards = 0
reward_temp = 0
while total_rewards < 9700:
    indx = np.random.randint(0, train_data01.shape[0])
    act = train_labels[indx]
    act = dqn.get_qs(train_data01[indx].reshape(1, -1))  #모델로 액션결정하기

    if np.random.rand() > epsilon: #랜덤으로 앱실론보다 크면
        act = np.argmax(act) #둘중 최대값 선택해서 진행
    else:
        act = np.random.randint(0, high=2) #무작위 탐험 걸리면 랜덤 액션
    r = dqn.play(train_data01[indx], action=act)

    dqn.update_replay(train_data01[indx], act, train_labels[indx], r)
    dqn.train()
    total_rewards = total_rewards + r

    if c%100 == 0 :
        print(c,"번째 에피소드 100까지의 총합 리워드 : ",total_rewards)
        total_rewards = 0

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

#%% 테스트 진행 (전처리)
test01 = test  # 백업

test01 = test01.drop(["Name"], axis=1)
test01 = test01.drop(["PassengerId"], axis=1)
test01 = test01.drop(["Ticket"], axis=1)
test01 = test01.drop(["Cabin"], axis=1)  # 전처리귀찮은거 다버려

# 결측치 대체
test01['Age'] = test01['Age'].fillna(0)
test01['Embarked'] = test01['Embarked'].fillna('S')
# 정수로 전처리
test01 = pd.get_dummies(data=test01, columns=['Embarked'], prefix='Embarked')
test01 = pd.get_dummies(data=test01, columns=['Sex'], prefix='Sex')
# 2차 백업
test02 = test01

test02 = (test02 - test02.mean(axis=0)) / test02.std(axis=0)

test03 = pd.DataFrame.to_numpy(test02)
#%%
test_results = np.argmax(dqn.model.predict(test02, batch_size=BATCH_SIZE), axis=1)
test_results
#%% csv파일 만들기
PassengerId =np.array(test["PassengerId"]).astype(int)
solution = pd.DataFrame(test_results, PassengerId, columns = ["Survived"])
print(solution.shape)
#%%
solution.to_csv("solution_forest.csv", index_label = ["PassengerId"])