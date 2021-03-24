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


input4 = pd.read_csv(r'E:\学习\太原理工大学\课题\test\four\FourFeature.csv', header=None,
                          low_memory=False)
output4 = pd.read_csv(r'E:\学习\太原理工大学\课题\test\four\FourLabels.csv', header=None,
                           low_memory=False)




train_input4 = input4.iloc[:, :12300]
train_output4 = output4.iloc[:,:]
print(train_input4.shape)
print(train_output4.shape)
# test_input4 = input4.iloc[1672:, :9225]
# test_output4 = output4.iloc[1672:,:]
# print(test_input4.shape)
# print(test_output4.shape)
print(train_output4.max() - train_output4.min())
print(train_output4.min())
# train_output4 = (train_output4 - train_output4.min()) / (train_output4.max() - train_output4.min())

# test_output4 = (test_output4 - test_output4.min()) / (test_output4.max() - test_output4.min())
#
train_input_list4 = [train_input4]
train_input_list4 = np.concatenate(train_input_list4, axis=0)
train_input_list4 = train_input_list4.reshape(1, 2094, 12300)
train_output_list4 = [train_output4]
train_output_list4 = np.concatenate(train_output_list4, axis=0)
train_output_list4 = train_output_list4.reshape(1, 2094, 16)

# test_input_list4 = [test_input4]
# test_input_list4 = np.concatenate(test_input_list4, axis=0)
# test_input_list4 = test_input_list4.reshape(1, 418, 9225)
# test_output_list4 = [test_output4]
# test_output_list4 = np.concatenate(test_output_list4, axis=0)
# test_output_list4 = test_output_list4.reshape(1, 418, 12)

model4 = tf.keras.Sequential()

forwardLayer = tf.keras.layers.LSTM(64, return_sequences=True)
backwardLayer = tf.keras.layers.LSTM(64, return_sequences=True,go_backwards=True)
model4.add(tf.keras.layers.Bidirectional(forwardLayer,backward_layer=backwardLayer,input_shape=(None,12300)))

model4.add(tf.keras.layers.Dense(1024, activation='sigmoid'))
model4.add(tf.keras.layers.Dropout(0.2))
model4.add(tf.keras.layers.Dense(256, activation='sigmoid'))
model4.add(tf.keras.layers.Dropout(0.2))
# model4.add(tf.keras.layers.Dense(42, activation='sigmoid'))
# model4.add(tf.keras.layers.Dropout(0.2))
model4.add(tf.keras.layers.Dense(16, activation='swish'))


model4.summary()

model4.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss="mae")

history4 = model4.fit(train_input_list4, train_output_list4, epochs=10000)
model4.save(r"E:\PyCharm_Projects\TensorFlow\Query\models\MyFourQueryModel.h5")
plt.plot(history4.epoch,history4.history.get("loss"),label="loss")
plt.legend();
plt.show()