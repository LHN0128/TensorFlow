# 迭代1000次

import tensorflow as tf
import numpy as np
import pandas as pd
# import keras
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Flatten
# import matplotlib.pyplot as plt

train_input = pd.read_csv(r'D:\WeChat\Documents\WeChat Files\wxid_1374843747513\FileStorage\File\2020-12\2190.csv', header=None,
                          low_memory=False)
train_output = pd.read_csv(r'D:\WeChat\Documents\WeChat Files\wxid_1374843747513\FileStorage\File\2020-12\2190label.csv', header=None,
                           low_memory=False)
# test_input = pd.read_csv(r'D:\zbh\java\experimentData\DataSet2result\MPL5\2190-546\1645_2190.csv', header=None,
#                          low_memory=False)
# test_output = pd.read_csv(r'D:\zbh\java\experimentData\DataSet2result\MPL5\2190-546\Lable_1645_2190.csv', header=None,
#                           low_memory=False)
print(train_input.shape)
print(train_output.shape)
# print(test_input.shape)
# print(test_output.shape)
train_input_list = [train_input]
# test_input_list = [test_input]
# train_input_list=np.array(train_input_list)
train_input_list = np.concatenate(train_input_list, axis=0)
print(train_input_list.shape)
# test_input_list = np.concatenate(test_input_list, axis=0)
# print(test_input_list.shape)
train_input_list = train_input_list.reshape(1, 2190, 9225)
print(train_input_list.shape)
# test_input_list = test_input_list.reshape(1, 546, 15775)
# train_output = train_output
train_output_list = [train_output]
train_output_list = np.concatenate(train_output_list, axis=0)
print(train_output_list.shape)
# test_output_list = [test_output]
# test_output_list = np.concatenate(test_output_list, axis=0)

train_output_list = train_output_list.reshape(1, 2190, 3)
print(train_output_list.shape)

# test_output_list = test_output_list.reshape(1, 546, 3)
model = tf.keras.Sequential()

model.add(tf.keras.layers.LSTM(64, input_shape=(None, 9225), return_sequences=True))  # 调参
# model.add(tf.keras.layers.Dropout(0.5))
# stateful = True,
# recurrent_initializer = 'glorot_uniform'))
# return_sequences = True))
# model.add(tf.keras.layers.Dense(50,activation='relu'))  #尝试去掉这一层
# model.add(tf.keras.layers.Dropout(0.8))
# model.add(tf.keras.layers.LSTM((layers[2],return_sequences=False))
model.add(tf.keras.layers.Dense(1000, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(3, activation='sigmoid'))

model.summary()

# model.summary()
# model.compile(optimizer = 'adam', loss = 'binary_crossentropy',metrics = ['accuracy'])


model.compile(optimizer='adam', loss=tf.keras.losses.categorical_crossentropy, metrics=["acc"])
# train_input_list = train_input_list.astype('float64')
# train_output_list = train_output_list.astype('float64')
history = model.fit(train_input_list, train_output_list, epochs=1000
                    )