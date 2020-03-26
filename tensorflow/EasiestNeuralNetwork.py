import tensorflow as tf
import numpy as np
from tensorflow import keras
import pandas
# tf.keras是官方的核心API
#定义模型
model = tf.keras.Sequential(
            [keras.layers.Dense(units=1, input_shape=[1])]
            )
model.compile(optimizer='sgd',
            loss='mean_squared_error')
#创造训练集，用于拟合y=2x-1
xs = np.array([-1.0,  0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0, 9.0], dtype=float)

model.fit(xs, ys, epochs=1000)
print(model.predict([20.0]))