import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


((train_image ,train_label),(test_image, test_label)) = tf.keras.datasets.mnist.load_data()

print(train_image.shape,train_label.shape,"    ", test_image.shape,test_label.shape)
# 显示某一个图像
plt.imshow(test_image[0])
plt.show()
# 用数值来分类
print(train_label)
#训练之前将数据归一化，把从0-255改为从0-1.缩小数据的范围，更容易收敛
train_image = train_image/255
test_image = test_image/255

# 优化，使用One-hot编码，可以从顺序编码直接转换。
train_label_onehot = tf.keras.utils.to_categorical(train_label)
test_label_onehot = tf.keras.utils.to_categorical(test_label)


model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
#两个隐藏层
model.add(tf.keras.layers.Dense(128,activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(128,activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
#对于多分类问题，输出层使用softmax来激活
model.add(tf.keras.layers.Dense(10,activation="softmax"))

#编译模型，顺序编码使用sparse_categorical_crossentropy，one-hot编码使用categorical_crossentropy
# model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['acc'])
model.compile(optimizer="Adam",loss='categorical_crossentropy',metrics=['acc'])

# 模型训练
# model.fit(train_image,train_label,epochs=5)
history = model.fit(train_image,train_label_onehot,epochs=5,validation_data=(test_image,test_label_onehot))
plt.plot(history.epoch,history.history.get("acc"),label="acc")
plt.plot(history.epoch,history.history.get("val_acc"),label="val_acc")
plt.legend()
plt.show()
#评估模型
# model.evaluate(test_image,test_label)
model.evaluate(test_image,test_label_onehot)


#用测试集预测一个
predict = model.predict(test_image)
print('预测值：',np.argmax(predict[0]),"   真实值：",test_label[0])
# print(predict)
model.summary()
