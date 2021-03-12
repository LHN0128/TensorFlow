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


input3 = pd.read_csv(r'E:\学习\太原理工大学\课题\test\three3\threeFCPFQI.csv', header=None,
                          low_memory=False)
output3 = pd.read_csv(r'E:\学习\太原理工大学\课题\test\three3\ThreeFinalLabels.csv', header=None,
                           low_memory=False)




train_input3 = input3.iloc[:1672, :]
train_output3 = output3.iloc[:1672,:]
test_input3 = input3.iloc[1672:, :]
test_output3 = output3.iloc[1672:,:]
train_output3 = (train_output3 - train_output3.min()) / (train_output3.max() - train_output3.min())
test_output3 = (test_output3 - test_output3.min()) / (test_output3.max() - test_output3.min())

train_input_list3 = [train_input3]
train_input_list3 = np.concatenate(train_input_list3, axis=0)
train_input_list3 = train_input_list3.reshape(1, 1672, 16845)
train_output_list3 = [train_output3]
train_output_list3 = np.concatenate(train_output_list3, axis=0)
train_output_list3 = train_output_list3.reshape(1, 1672, 3)

test_input_list3 = [test_input3]
test_input_list3 = np.concatenate(test_input_list3, axis=0)
test_input_list3 = test_input_list3.reshape(1, 418, 16845)
test_output_list3 = [test_output3]
test_output_list3 = np.concatenate(test_output_list3, axis=0)
test_output_list3 = test_output_list3.reshape(1, 418, 3)

model3 = tf.keras.Sequential()

forwardLayer = tf.keras.layers.LSTM(64, return_sequences=True)
backwardLayer = tf.keras.layers.LSTM(64, return_sequences=True,go_backwards=True)
model3.add(tf.keras.layers.Bidirectional(forwardLayer,backward_layer=backwardLayer,merge_mode="concat", input_shape=(None, 16845)))
model3.add(tf.keras.layers.Dense(256, activation='swish'))
model3.add(tf.keras.layers.Dropout(0.5))
model3.add(tf.keras.layers.Dense(64, activation='swish'))
model3.add(tf.keras.layers.Dropout(0.5))
model3.add(tf.keras.layers.Dense(3, activation='softmax'))


model3.summary()

# model.summary()
# model.compile(optimizer = 'adam', loss = 'binary_crossentropy',metrics = ['accuracy'])


model3.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss=tf.keras.losses.categorical_crossentropy, metrics=["acc"])
# train_input_list = train_input_list.astype('float64')
# train_output_list = train_output_list.astype('float64')
history3 = model3.fit(train_input_list3, train_output_list3,validation_data=(test_input_list3,test_output_list3), epochs=150)



plt.plot(history3.epoch,history3.history.get("val_acc"),label="MPL=3",color="b")


plt.title("MPL = 3时 测试集精度")
plt.legend();
plt.show();
