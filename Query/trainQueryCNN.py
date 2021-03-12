import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


train_input = pd.read_csv(r'E:\学习\太原理工大学\课题\test\two\planFeatures\ConbinedFeatures.csv',header=None, low_memory=False)
train_input = train_input.iloc[:, :6150]
train_output = pd.read_csv(r'E:\学习\太原理工大学\课题\test\two\TwoLabels.csv',header=None, low_memory=False)
train_output = (train_output - train_output.min()) / (train_output.max() - train_output.min())

model = tf.keras.Sequential()
# model.add(tf.keras.layers.Embedding(120,120,input_length=6150))#把每个词变为120的向量，
model.add(tf.keras.layers.Conv1D(32,7,activation="relu"))#（单元数，卷积核大小，激活函数）
model.add(tf.keras.layers.MaxPooling1D(2))
model.add(tf.keras.layers.Dense(64,activation='relu'))
model.add(tf.keras.layers.Dense(8,activation='sigmoid'))
model.summary()




model.compile(optimizer=tf.keras.optimizers.Adam(0.0003),loss=tf.keras.losses.mse,metrics=tf.keras.metrics.mse)
history = model.fit(train_input,train_output,epochs=100)
plt.plot(history.epoch,history.history.get("loss"),label="loss")
plt.legend();
plt.show();