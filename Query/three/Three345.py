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





input4 = pd.read_csv(r'E:\学习\太原理工大学\课题\test\three4\threeFCPFQI.csv', header=None,
                          low_memory=False)
output4 = pd.read_csv(r'E:\学习\太原理工大学\课题\test\three4\ThreeFinalLabels.csv', header=None,
                           low_memory=False)




train_input4 = input4.iloc[:1672, :]
train_output4 = output4.iloc[:1672,:]
test_input4 = input4.iloc[1672:, :]
test_output4 = output4.iloc[1672:,:]
train_output4 = (train_output4 - train_output4.min()) / (train_output4.max() - train_output4.min())
test_output4 = (test_output4 - test_output4.min()) / (test_output4.max() - test_output4.min())

train_input_list4 = [train_input4]
train_input_list4 = np.concatenate(train_input_list4, axis=0)
train_input_list4 = train_input_list4.reshape(1, 1672, 16845)
train_output_list4 = [train_output4]
train_output_list4 = np.concatenate(train_output_list4, axis=0)
train_output_list4 = train_output_list4.reshape(1, 1672, 3)

test_input_list4 = [test_input4]
test_input_list4 = np.concatenate(test_input_list4, axis=0)
test_input_list4 = test_input_list4.reshape(1, 418, 16845)
test_output_list4 = [test_output4]
test_output_list4 = np.concatenate(test_output_list4, axis=0)
test_output_list4 = test_output_list4.reshape(1, 418, 3)

model4 = tf.keras.Sequential()

forwardLayer = tf.keras.layers.LSTM(64, return_sequences=True)
backwardLayer = tf.keras.layers.LSTM(64, return_sequences=True,go_backwards=True)
model4.add(tf.keras.layers.Bidirectional(forwardLayer,backward_layer=backwardLayer,merge_mode="concat", input_shape=(None, 16845)))
model4.add(tf.keras.layers.Dense(256, activation='swish'))
model4.add(tf.keras.layers.Dropout(0.5))
model4.add(tf.keras.layers.Dense(64, activation='swish'))
model4.add(tf.keras.layers.Dropout(0.5))
model4.add(tf.keras.layers.Dense(3, activation='softmax'))


model4.summary()

# model.summary()
# model.compile(optimizer = 'adam', loss = 'binary_crossentropy',metrics = ['accuracy'])


model4.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss=tf.keras.losses.categorical_crossentropy, metrics=["acc"])
# train_input_list = train_input_list.astype('float64')
# train_output_list = train_output_list.astype('float64')
history4 = model4.fit(train_input_list4, train_output_list4,validation_data=(test_input_list4,test_output_list4), epochs=150)












input5 = pd.read_csv(r'E:\学习\太原理工大学\课题\test\three5\threeFCPFQI.csv', header=None,
                          low_memory=False)
output5 = pd.read_csv(r'E:\学习\太原理工大学\课题\test\three5\ThreeFinalLabels.csv', header=None,
                           low_memory=False)




train_input5= input5.iloc[:1672, :]
train_output5 = output5.iloc[:1672,:]
test_input5 = input5.iloc[1672:, :]
test_output5 = output5.iloc[1672:,:]
train_output5 = (train_output5 - train_output5.min()) / (train_output5.max() - train_output5.min())
test_output5 = (test_output5 - test_output5.min()) / (test_output5.max() - test_output5.min())

train_input_list5 = [train_input5]
train_input_list5 = np.concatenate(train_input_list5, axis=0)
train_input_list5 = train_input_list5.reshape(1, 1672, 16845)
train_output_list5 = [train_output5]
train_output_list5 = np.concatenate(train_output_list5, axis=0)
train_output_list5 = train_output_list5.reshape(1, 1672, 3)

test_input_list5 = [test_input5]
test_input_list5 = np.concatenate(test_input_list5, axis=0)
test_input_list5 = test_input_list5.reshape(1, 418, 16845)
test_output_list5 = [test_output5]
test_output_list5 = np.concatenate(test_output_list5, axis=0)
test_output_list5 = test_output_list5.reshape(1, 418, 3)

model5 = tf.keras.Sequential()

forwardLayer = tf.keras.layers.LSTM(64, return_sequences=True)
backwardLayer = tf.keras.layers.LSTM(64, return_sequences=True,go_backwards=True)
model5.add(tf.keras.layers.Bidirectional(forwardLayer,backward_layer=backwardLayer,merge_mode="concat", input_shape=(None, 16845)))
model5.add(tf.keras.layers.Dense(256, activation='swish'))
model5.add(tf.keras.layers.Dropout(0.5))
model5.add(tf.keras.layers.Dense(64, activation='swish'))
model5.add(tf.keras.layers.Dropout(0.5))
model5.add(tf.keras.layers.Dense(3, activation='softmax'))


model5.summary()

# model.summary()
# model.compile(optimizer = 'adam', loss = 'binary_crossentropy',metrics = ['accuracy'])


model5.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss=tf.keras.losses.categorical_crossentropy, metrics=["acc"])
# train_input_list = train_input_list.astype('float64')
# train_output_list = train_output_list.astype('float64')
history5 = model5.fit(train_input_list5, train_output_list5,validation_data=(test_input_list5,test_output_list5), epochs=150)





plt.plot(history3.epoch,history3.history.get("val_acc"),label="MPL=3",color="b")
plt.plot(history4.epoch,history4.history.get("val_acc"),label="MPL=4",color="r")
plt.plot(history5.epoch,history5.history.get("val_acc"),label="MPL=5",color="g")

plt.title("MPL = 3、4、5时 测试集精度")
plt.legend();
plt.show();
