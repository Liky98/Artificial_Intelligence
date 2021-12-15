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

train_data, test_data, train_labels, test_labels = train_test_split(train_data, labels, test_size=0.3,random_state=True)
train_data01 = pd.DataFrame.to_numpy(train_data)
test_data01 =  pd.DataFrame.to_numpy(test_data)

input_size = train_data01.shape[1]
output_size = 1


#%%
from keras.models import Sequential
from keras.layers import Dense, Activation, BatchNormalization, Dropout
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam

from keras.layers import Input
import keras
from keras.models import Model


def DenseNet(X_train):
    ip = Input(shape=(X_train.shape[1],))
    x_list = [ip]

    x = Dense(64, use_bias=False)(ip)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.1)(x)

    x_list.append(x)
    x = keras.layers.concatenate(x_list)
    x = Dense(32, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.1)(x)

    x_list.append(x)
    x = keras.layers.concatenate(x_list)
    x = Dense(16, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.1)(x)

    x_list.append(x)
    x = keras.layers.concatenate(x_list)
    x = Dense(8, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.1)(x)

    x_list.append(x)
    x = keras.layers.concatenate(x_list)
    x = Dense(4, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.1)(x)

    op = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=ip, outputs=op)
    adam = Adam(lr=0.05 )
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
    return model


model = DenseNet(train_data01)
model.summary()
#%%
history_model = model.fit(train_data01, train_labels, epochs=1200, batch_size=200, verbose=0,
                          validation_split=0.1)

scores = model.evaluate(train_data01, train_labels, verbose=0)
print("Train %s: %.3f%%" % (model.metrics_names[1], scores[1] * 100))

scores = model.evaluate(test_data01, test_labels, verbose=0)
print("Validation %s: %.3f%%" % (model.metrics_names[1], scores[1] * 100))

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
test_results = np.argmax(model.predict(test03, batch_size=32), axis=1)
test_results
#%% csv파일 만들기
PassengerId =np.array(test["PassengerId"]).astype(int)
solution = pd.DataFrame(test_results, PassengerId, columns = ["Survived"])
print(solution.shape)
#%%
solution.to_csv("only_Dense1121.csv", index_label = ["PassengerId"])