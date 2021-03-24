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


input3 = pd.read_csv(r'E:\学习\太原理工大学\课题\test\three3\ThreeFeature.csv', header=None,
                          low_memory=False)
output3 = pd.read_csv(r'E:\学习\太原理工大学\课题\test\three3\ThreeLabels.csv', header=None,
                           low_memory=False)




train_input3 = input3.iloc[:, :9225]
train_output3 = output3.iloc[:,:]
print(train_input3.shape)
print(train_output3.shape)
# test_input3 = input3.iloc[1672:, :9225]
# test_output3 = output3.iloc[1672:,:]
# print(test_input3.shape)
# print(test_output3.shape)
print(train_output3.max() - train_output3.min())
print(train_output3.min())
train_output3 = (train_output3 - train_output3.min()) / (train_output3.max() - train_output3.min())

# test_output3 = (test_output3 - test_output3.min()) / (test_output3.max() - test_output3.min())
#
train_input_list3 = [train_input3]
train_input_list3 = np.concatenate(train_input_list3, axis=0)
train_input_list3 = train_input_list3.reshape(1, 2090, 9225)
train_output_list3 = [train_output3]
train_output_list3 = np.concatenate(train_output_list3, axis=0)
train_output_list3 = train_output_list3.reshape(1, 2090, 12)

# test_input_list3 = [test_input3]
# test_input_list3 = np.concatenate(test_input_list3, axis=0)
# test_input_list3 = test_input_list3.reshape(1, 418, 9225)
# test_output_list3 = [test_output3]
# test_output_list3 = np.concatenate(test_output_list3, axis=0)
# test_output_list3 = test_output_list3.reshape(1, 418, 12)

model3 = tf.keras.Sequential()

forwardLayer = tf.keras.layers.LSTM(64, return_sequences=True)
backwardLayer = tf.keras.layers.LSTM(64, return_sequences=True,go_backwards=True)
model3.add(tf.keras.layers.Bidirectional(forwardLayer,backward_layer=backwardLayer,input_shape=(None,9225)))

model3.add(tf.keras.layers.Dense(512, activation='sigmoid'))
model3.add(tf.keras.layers.Dropout(0.2))
model3.add(tf.keras.layers.Dense(128, activation='sigmoid'))
model3.add(tf.keras.layers.Dropout(0.2))
# model3.add(tf.keras.layers.Dense(32, activation='sigmoid'))
# model3.add(tf.keras.layers.Dropout(0.2))
model3.add(tf.keras.layers.Dense(12, activation='swish'))


model3.summary()

model3.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss="mae")

history3 = model3.fit(train_input_list3, train_output_list3, epochs=300)
model3.save(r"E:\PyCharm_Projects\TensorFlow\Query\models\MyThreeQueryModel.h5")
plt.plot(history3.epoch,history3.history.get("loss"),label="loss")
plt.legend();
plt.show()