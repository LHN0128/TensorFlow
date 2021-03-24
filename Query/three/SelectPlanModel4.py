#本文件构建模型，用来选择执行计划
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#无视下述警告即可
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession


config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


input = pd.read_csv(r'E:\学习\太原理工大学\课题\test\three4\threeFCPFQI.csv', header=None,
                          low_memory=False)
output = pd.read_csv(r'E:\学习\太原理工大学\课题\test\three4\ThreeFinalLabels.csv', header=None,
                           low_memory=False)




train_input = input.iloc[:1672, :]
train_output = output.iloc[:1672,:]
test_input = input.iloc[1672:, :]
test_output = output.iloc[1672:,:]
train_output = (train_output - train_output.min()) / (train_output.max() - train_output.min())
test_output = (test_output - test_output.min()) / (test_output.max() - test_output.min())

train_input_list = [train_input]
train_input_list = np.concatenate(train_input_list, axis=0)
train_input_list = train_input_list.reshape(1, 1672, 16845)
train_output_list = [train_output]
train_output_list = np.concatenate(train_output_list, axis=0)
train_output_list = train_output_list.reshape(1, 1672, 3)

test_input_list = [test_input]
test_input_list = np.concatenate(test_input_list, axis=0)
test_input_list = test_input_list.reshape(1, 418, 16845)
test_output_list = [test_output]
test_output_list = np.concatenate(test_output_list, axis=0)
test_output_list = test_output_list.reshape(1, 418, 3)

model = tf.keras.Sequential()

forwardLayer = tf.keras.layers.LSTM(64, return_sequences=True)
backwardLayer = tf.keras.layers.LSTM(64, return_sequences=True,go_backwards=True)
model.add(tf.keras.layers.Bidirectional(forwardLayer,backward_layer=backwardLayer,merge_mode="concat", input_shape=(None, 16845)))
model.add(tf.keras.layers.Dense(256, activation='swish'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(64, activation='swish'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(3, activation='softmax'))


model.summary()

# model.summary()
# model.compile(optimizer = 'adam', loss = 'binary_crossentropy',metrics = ['accuracy'])


model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss=tf.keras.losses.categorical_crossentropy, metrics=["acc"])
# train_input_list = train_input_list.astype('float64')
# train_output_list = train_output_list.astype('float64')
history = model.fit(train_input_list, train_output_list,validation_data=(test_input_list,test_output_list), epochs=150)

plt.plot(history.epoch,history.history.get("acc"),label="acc")
plt.plot(history.epoch,history.history.get("val_acc"),label="val_acc")
plt.legend();
plt.show();
