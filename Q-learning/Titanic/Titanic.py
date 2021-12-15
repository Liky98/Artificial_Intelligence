import os
import numpy as np
import tensorflow_datasets as tfds
from imbDRL.agents.ddqn import TrainDDQN
from imbDRL.data import get_train_test_val
from imbDRL.utils import rounded_dict
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
from tensorflow.keras import Model
import pandas as pd


episodes = 16_000  #
warmup_steps = 16_000
memory_length = warmup_steps
batch_size = 32
collect_steps_per_episode = 500
collect_every = 500

target_update_period = 400
target_update_tau = 1
n_step_update = 1

learning_rate = 0.00025
gamma = 0.0
min_epsilon = 0.5
decay_episodes = episodes // 10

min_class = [1]
maj_class = [0]

""" 모델 """
def Network_Model(input):
    output = LSTM(256, dropout=0.1, return_sequences=True, stateful=False, kernel_initializer='random_normal')(input)
    output = BatchNormalization()(output)
    output = LSTM(128, dropout=0.1, return_sequences=True, stateful=False, kernel_initializer='random_normal')(
        output)
    output = BatchNormalization()(output)
    output = LSTM(64, dropout=0.1, return_sequences=True, stateful=False, kernel_initializer='random_normal')(
        output)
    output = BatchNormalization()(output)
    output = LSTM(32, dropout=0.1, stateful=False, kernel_initializer='random_normal')(output)
    output = BatchNormalization()(output)
    return Model(input, output)

layers = [Dense(256, activation="relu"), Dropout(0.2),
          Dense(256, activation="relu"), Dropout(0.2),
          Dense(2, activation=None)]  # No activation, pure Q-values

""" 데이터 불러오기 """
train = pd.read_csv('C:/Users/LeeKihoon/PycharmProjects/gachonProject/Q-learning/Titanic/train.csv')
test = pd.read_csv('C:/Users/LeeKihoon/PycharmProjects/gachonProject/Q-learning/Titanic/test.csv')

y = train['Survived']

#%%
df = tfds.as_dataframe(*tfds.load("titanic", split='train', with_info=True))
y = df.survived.values
df.columns
#%%
df = df.drop(columns=["survived", "features/boat", "features/cabin", "features/home.dest", "features/name", "features/ticket"])
df = df.astype(np.float64)
df = (df - df.min()) / (df.max() - df.min())  # 정규화는 열차와 테스트 세트를 분할한 후에 이루어짐
#%%

X_train, X_test, y_train, y_test = train_test_split(df.to_numpy(), y, stratify=y, test_size=0.2)

X_train, y_train, X_test, y_test, X_val, y_val = get_train_test_val(X_train, y_train, X_test, y_test,
                                                                    min_class, maj_class, val_frac=0.2)
#%%
model = TrainDDQN(episodes, warmup_steps, learning_rate, gamma, min_epsilon, decay_episodes, target_update_period=target_update_period,
                  target_update_tau=target_update_tau, batch_size=batch_size, collect_steps_per_episode=collect_steps_per_episode,
                  memory_length=memory_length, collect_every=collect_every, n_step_update=n_step_update)

#%%
model.compile_model(X_train, y_train, layers)
model.q_net.summary()
model.train(X_val, y_val)
#%%
stats = model.evaluate(X_test, y_test, X_train, y_train)
#%%
print(rounded_dict(stats))
#%%


