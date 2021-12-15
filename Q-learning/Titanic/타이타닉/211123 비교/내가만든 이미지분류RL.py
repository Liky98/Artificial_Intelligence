import os
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

path = "기계학습프로그래밍/img/Fruit/"
print("싱싱한 귤 Train 데이터 량 : ", len(os.listdir(path+"train/fresh")))
print("썪은 귤 Train 데이터 량 : ", len(os.listdir(path+"train/rotten")))
print("싱싱한 귤 validation 데이터 량 : ", len(os.listdir(path+"validation/fresh")))
print("썪은 귤 validation 데이터 량 : ", len(os.listdir(path+"validation/rotten")))
print("싱싱한 귤 Test 데이터 량 : ", len(os.listdir(path+"test/fresh")))
print("썪은 귤 Test 데이터 량 : ", len(os.listdir(path+"test/rotten")))

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    path+"train", target_size=(150,150), batch_size=20, class_mode='binary'
)
validation_generator = test_datagen.flow_from_directory(
    path+"validation", target_size=(150,150), batch_size=20, class_mode='binary'
)

def create_model(self, input_size, output_size):
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

    model = models.Sequential()
    model.add(pretrainingModel_inception_v3)
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(2, activation=None))

    ad = optimizers.Adam(learning_rate=0.001)  # 'rmsprop'
    model.compile(optimizer=ad,
                  loss='mse', metrics=['mae'])
    history = model.fit_generator(
        train_generator, steps_per_epoch=100, epochs=30, validation_data=validation_generator, validation_steps=50
    )
    return model

def action() :

def get_reward(action) :
