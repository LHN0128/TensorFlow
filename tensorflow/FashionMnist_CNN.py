import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

print(tf.test.is_gpu_available())

((train_image ,train_label),(test_image, test_label)) = tf.keras.datasets.mnist.load_data()
print(train_image.shape)
print(train_label.shape)
#对图像扩充维度处理，添加一个维度，最后一个维度为通道数
train_image = np.expand_dims(train_image,-1)
test_image = np.expand_dims(test_image,-1)

model = tf.keras.Sequential()


#conv2D(通道数,(卷积核大小)),padding使用same会保持卷积后大小和之前相同
#如果padding使用valid会让卷积后宽度变得更小
#隐藏单元数不要太小，防止信息瓶颈.提高网络深度，可以连续多个conv层
#注意卷积核大小是2的n次方，提高拟合能力
model.add(tf.keras.layers.Conv2D(64,(3,3),input_shape=(28,28,1),activation="relu",padding="same"))
model.add(tf.keras.layers.Conv2D(64,(3,3),activation="relu",padding="same"))
#最大池化，缩小图片大小为原来一半
model.add(tf.keras.layers.MaxPool2D())
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Conv2D(128,(3,3),activation="relu",padding="same"))
model.add(tf.keras.layers.Conv2D(128,(3,3),activation="relu",padding="same"))
model.add(tf.keras.layers.MaxPool2D())
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Conv2D(256,(3,3),activation="relu",padding="same"))
model.add(tf.keras.layers.Conv2D(256,(3,3),activation="relu",padding="same"))
model.add(tf.keras.layers.MaxPool2D())
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Conv2D(512,(3,3),activation="relu",padding="same"))
model.add(tf.keras.layers.Conv2D(512,(3,3),activation="relu",padding="same"))
# #全局平均池化，目前维度为(None, 12, 12, 64)
# 但是全局平均池化可以把其他维度去掉只保留通道数维度，然后就可以添加全连接层
model.add(tf.keras.layers.GlobalAveragePooling2D())
model.add(tf.keras.layers.Dense(512,activation="relu"))
model.add(tf.keras.layers.Dense(10,activation="softmax"))


model.summary()

model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["acc"])
history = model.fit(train_image,train_label,epochs=30,validation_data=(test_image,test_label))

plt.plot(history.epoch,history.history.get("acc"),label="acc")
plt.plot(history.epoch,history.history.get("val_acc"),label="val_acc")
plt.plot(history.epoch,history.history.get("loss"),label="loss")
plt.plot(history.epoch,history.history.get("val_loss"),label="val_loss")
plt.legend()
plt.show()
