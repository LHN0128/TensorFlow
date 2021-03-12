import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

((train_image ,train_label),(test_image, test_label)) = tf.keras.datasets.fashion_mnist.load_data()

# print(train_image.shape,train_label.shape,"    ", test_image.shape,test_label.shape)
# 显示某一个图像
# plt.imshow(train_image[0])
# plt.show()
# 用数值来分类
print(train_label)
print("test_image_shape:",test_image.shape)
#训练之前将数据归一化，把从0-255改为从0-1
train_image = train_image/255
test_image = test_image/255
# 优化，使用One-hot编码，可以从顺序编码直接转换。
train_label_onehot = tf.keras.utils.to_categorical(train_label)
test_label_onehot = tf.keras.utils.to_categorical(test_label)


# 使用函数式调用来创建模型
input = tf.keras.Input(shape=(28,28))
x = tf.keras.layers.Flatten()(input)
x = tf.keras.layers.Dense(128,activation='relu')(x)
x = tf.keras.layers.Dropout(0.5)(x)
x = tf.keras.layers.Dense(128,activation='relu')(x)
x = tf.keras.layers.Dropout(0.5)(x)
output = tf.keras.layers.Dense(10,activation="softmax")(x)
model = tf.keras.Model(inputs=input,outputs=output)


model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),loss='categorical_crossentropy',metrics=['acc'])
#
# # 模型训练
# # 在训练过程中还提示测试数据的变化，使用validation_data
history = model.fit(train_image,train_label_onehot,epochs=1000,validation_data=(test_image,test_label_onehot))
#
# print(history.history.keys())
#
#
# 绘制loss和val_loss的图像，发现val_loss中途升高，说明过拟合
#测试数据上的得分acc低于训练数据，过拟合
plt.plot(history.epoch,history.history.get("loss"),label="loss")
plt.plot(history.epoch,history.history.get("val_loss"),label="val_loss")
plt.legend()#显示图例
plt.show()


# 绘制acc和val_acc的图像，发现val_acc中途升高，说明过拟合
#测试数据上的得分acc低于训练数据，过拟合
plt.plot(history.epoch,history.history.get("acc"),label="acc")
plt.plot(history.epoch,history.history.get("val_acc"),label="val_acc")
plt.legend()#显示图例
plt.show()

#评估模型
model.evaluate(test_image,test_label_onehot)



#用测试集预测一个
predict = model.predict(test_image)
print('预测值：',np.argmax(predict[0]),"   真实值：",test_label[0])

model.summary()
