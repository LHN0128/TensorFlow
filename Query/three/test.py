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


model3 = tf.keras.Sequential()

forwardLayer = tf.keras.layers.LSTM(64, return_sequences=True)
backwardLayer = tf.keras.layers.LSTM(64, return_sequences=True,go_backwards=True)
model3.add(tf.keras.layers.Bidirectional(forwardLayer,backward_layer=backwardLayer,merge_mode="concat", input_shape=(None, 12300)))
model3.add(tf.keras.layers.Dense(256, activation='swish'))
model3.add(tf.keras.layers.Dropout(0.5))
model3.add(tf.keras.layers.Dense(64, activation='swish'))
model3.add(tf.keras.layers.Dropout(0.5))
model3.add(tf.keras.layers.Dense(3, activation='softmax'))

model3.summary()