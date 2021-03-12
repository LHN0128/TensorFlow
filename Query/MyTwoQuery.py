# 迭代1000次

import tensorflow as tf
import numpy as np
import pandas as pd

# import keras
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Flatten
import matplotlib.pyplot as plt


#无视下述警告即可
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


train_input = pd.read_csv(r'E:\学习\太原理工大学\课题\test\two\planFeatures\ConbinedFeatures.csv', header=None,
                          low_memory=False)
train_output = pd.read_csv(r'E:\学习\太原理工大学\课题\test\two\TwoLabels.csv', header=None,
                           low_memory=False)
# test_input = pd.read_csv(r'D:\zbh\java\experimentData\DataSet2result\MPL5\2190-546\1645_2190.csv', header=None,
#                          low_memory=False)
# test_output = pd.read_csv(r'D:\zbh\java\experimentData\DataSet2result\MPL5\2190-546\Lable_1645_2190.csv', header=None,
#                           low_memory=False)
print(train_input.shape)
print(train_output.shape)
# print(test_input.shape)
# print(test_output.shape)
#数据预处理
train_input = train_input.iloc[:, :6150]
train_output = (train_output - train_output.min()) / (train_output.max() - train_output.min())


train_input_list = [train_input]
# test_input_list = [test_input]
# train_input_list=np.array(train_input_list)
train_input_list = np.concatenate(train_input_list, axis=0)
print(train_input_list.shape)
# test_input_list = np.concatenate(test_input_list, axis=0)
# print(test_input_list.shape)
train_input_list = train_input_list.reshape(1, 666, 6150)
print(train_input_list.shape)
# test_input_list = test_input_list.reshape(1, 546, 15775)
# train_output = train_output
train_output_list = [train_output]
train_output_list = np.concatenate(train_output_list, axis=0)
print(train_output_list.shape)
# test_output_list = [test_output]
# test_output_list = np.concatenate(test_output_list, axis=0)

train_output_list = train_output_list.reshape(1, 666, 8)
print(train_output_list.shape)

# test_output_list = test_output_list.reshape(1, 546, 3)
model = tf.keras.Sequential()

# model = tf.keras.layers.LSTM(64, return_sequences=True)  # 调参
forwardLayer = tf.keras.layers.LSTM(64, return_sequences=True)
backwardLayer = tf.keras.layers.LSTM(64, return_sequences=True,go_backwards=True)
model.add(tf.keras.layers.Bidirectional(forwardLayer,backward_layer=backwardLayer, input_shape=(None, 6150)))
# model.add(tf.keras.layers.Dropout(0.5))
# stateful = True,
# recurrent_initializer = 'glorot_uniform'))
# return_sequences = True))
# model.add(tf.keras.layers.Dense(50,activation='relu'))  #尝试去掉这一层
# model.add(tf.keras.layers.Dropout(0.8))
# model.add(tf.keras.layers.LSTM((layers[2],return_sequences=False))
model.add(tf.keras.layers.Dense(1000, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(8, activation='sigmoid'))


model.summary()

# model.summary()
# model.compile(optimizer = 'adam', loss = 'binary_crossentropy',metrics = ['accuracy'])


model.compile(optimizer='adam', loss="mse", metrics=["mae"])
# train_input_list = train_input_list.astype('float64')
# train_output_list = train_output_list.astype('float64')
history = model.fit(train_input_list, train_output_list, epochs=1000
                    )

plt.plot(history.epoch,history.history.get("mae"),label="mae")
plt.plot(history.epoch,history.history.get("loss"),label="mse")
plt.legend();
plt.show();