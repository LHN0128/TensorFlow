import tensorflow as tf
import matplotlib.pyplot as plt


((train_images ,train_labels),(test_images, test_labels)) = tf.keras.datasets.fashion_mnist.load_data()
#数据归一化
train_images = train_images/255
test_images = test_images/255

ds_train_img = tf.data.Dataset.from_tensor_slices(train_images)
print(ds_train_img)
ds_train_lab = tf.data.Dataset.from_tensor_slices(train_labels)
print(ds_train_lab)#<TensorSliceDataset shapes: (), types: tf.uint8>,()代表一个数字，uint表示无符号整型

#将两个dataset对应在一起，合并为一个元组，训练集图片和标签对应
ds_train = tf.data.Dataset.zip((ds_train_img,ds_train_lab))
#或者直接取出就是元组的形式，省略zip那一步。test
ds_test = tf.data.Dataset.from_tensor_slices((test_images,test_labels))


#取出10000个组件乱序，重复（不需要告诉多少，可以无限制重复），batch_size为64,即步长为64。测试集数据不需要做shuffle和repeat
ds_train = ds_train.shuffle(10000).repeat().batch(64)
ds_test = ds_test.batch(64)
print(ds_test)
#创建模型
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")

])
model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["acc"])

steps_per_epochs = train_images.shape[0]//64#python的除法这里使用//才是整除，数学计算都是精确的，不是取整

history = model.fit(ds_train,epochs=5,
          steps_per_epoch=steps_per_epochs,
          validation_data=ds_test,
          validation_steps=10000//64
          )

plt.plot(history.epoch,history.history.get("acc"),label="acc")
plt.plot(history.epoch,history.history.get("val_acc"),label="val_acc")
plt.legend()
plt.show()

model.evaluate(ds_test)
