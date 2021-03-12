# 该文件将查询组合执行计划特征FEP作为输入，输出查询交互特征FQI。度量查询交互模型

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


input = pd.read_csv(r'E:\学习\太原理工大学\课题\test\three\ThreeFeature.csv', header=None,
                          low_memory=False)
output = pd.read_csv(r'E:\学习\太原理工大学\课题\test\three\ThreeLabels.csv', header=None,
                           low_memory=False)
# test_input = pd.read_csv(r'D:\zbh\java\experimentData\DataSet2result\MPL5\2190-546\1645_2190.csv', header=None,
#                          low_memory=False)
# test_output = pd.read_csv(r'D:\zbh\java\experimentData\DataSet2result\MPL5\2190-546\Lable_1645_2190.csv', header=None,
#                           low_memory=False)
print(input.shape)
print(output.shape)
# print(test_input.shape)
# print(test_output.shape)
#数据预处理,划分训练集1672，测试集418条数据
train_input = input.iloc[:1672, :9225]
train_output = output.iloc[:1672,:]
test_input = input.iloc[1672:, :9225]
test_output = output.iloc[1672:,:]
train_output = (train_output - train_output.min()) / (train_output.max() - train_output.min())
test_output = (test_output - test_output.min()) / (test_output.max() - test_output.min())

train_input_list = [train_input]
train_input_list = np.concatenate(train_input_list, axis=0)
train_input_list = train_input_list.reshape(1, 1672, 9225)
train_output_list = [train_output]
train_output_list = np.concatenate(train_output_list, axis=0)
train_output_list = train_output_list.reshape(1, 1672, 12)

test_input_list = [test_input]
test_input_list = np.concatenate(test_input_list, axis=0)
test_input_list = test_input_list.reshape(1, 418, 9225)
test_output_list = [test_output]
test_output_list = np.concatenate(test_output_list, axis=0)
test_output_list = test_output_list.reshape(1, 418, 12)

model = tf.keras.Sequential()

forwardLayer = tf.keras.layers.LSTM(64, return_sequences=True)
backwardLayer = tf.keras.layers.LSTM(64, return_sequences=True,go_backwards=True)
model.add(tf.keras.layers.Bidirectional(forwardLayer,backward_layer=backwardLayer, input_shape=(None, 9225)))

model.add(tf.keras.layers.Dense(1000, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(12, activation='sigmoid'))


model.summary()

# model.summary()
# model.compile(optimizer = 'adam', loss = 'binary_crossentropy',metrics = ['accuracy'])


model.compile(optimizer='adam', loss="mse", metrics=["mae"])
# train_input_list = train_input_list.astype('float64')
# train_output_list = train_output_list.astype('float64')
history = model.fit(train_input_list, train_output_list,validation_data=(test_input_list,test_output_list), epochs=200)
model.save(r"E:\PyCharm_Projects\TensorFlow\Query\models\MyThreeQueryModel.h5")
plt.plot(history.epoch,history.history.get("val_loss"),label="val_loss")
plt.plot(history.epoch,history.history.get("loss"),label="loss")
plt.legend();
plt.show();


