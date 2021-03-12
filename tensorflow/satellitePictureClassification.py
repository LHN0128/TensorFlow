import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import random
import IPython.display as display

data_dir = r"E:\PyCharm Projects\TensorFlow\2_class"
# 通过pathlib获取路径对象
data_root = pathlib.Path(data_dir)
for item in data_root.iterdir():
    print(item)
    # 获取所有图片路径,并存放在list中。这些路径都是windowsPath对象
all_image_path = list(data_root.glob('*/*'))
print(len(all_image_path))#1400
# 变为真正的字符串路径
all_image_path = [str(path) for path in all_image_path]
# 乱序处理图片
random.shuffle(all_image_path)
# 记录图片个数方便划分测试和训练集
image_count = len(all_image_path)
#获取所有目录名:airplane,lake,根据首字母排序
label_names = sorted(item.name for item in data_root.glob("*/"))
# 对标签编号处理{'airplane': 0, 'lake': 1},可适用于任何多类型的分类
label_to_index = dict((name,index) for index,name in enumerate(label_names))
# 得到所有图片的label,为[1和0的列表]
all_image_label = [label_to_index[pathlib.Path(p).parent.name] for p in all_image_path]

index_to_label = dict((v,k) for k,v in label_to_index.items())#{0: 'airplane', 1: 'lake'}



def load_preprocess_image(path):
    #读取路径
    img_path = path
    # tensorflow中读取文件,以二进制形式,再通过解码得到数据
    img_raw = tf.io.read_file(img_path)
    img_tensor = tf.image.decode_jpeg(img_raw,channels=3)  # (256, 256, 3),得到的这个图片
    img_tensor = tf.image.resize(img_tensor,[256,256])#将图片变形成设定的大小,不会影响训练
    # 转换数据类型,方可标准化
    img_tensor = tf.cast(img_tensor, tf.float32)
    # 标准化
    img = img_tensor / 255
    #返回处理好的图片数据
    return img

# image_path = all_image_path[120]
# plt.imshow(load_preprocess_image(image_path))
# plt.show()
#构造tf.dataset,加载所有图片和label
path_ds = tf.data.Dataset.from_tensor_slices(all_image_path)
image_dataset = path_ds.map(load_preprocess_image)
label_dataset = tf.data.Dataset.from_tensor_slices(all_image_label)
#将图片和标签合并为一个dataset方便训练
dataset = tf.data.Dataset.zip((image_dataset,label_dataset))
#划分训练集和测试集
test_count = int(image_count*0.2)
train_count = image_count-test_count
train_dataset = dataset.skip(test_count)#skip是跳过
test_dateset = dataset.take(test_count)#take是取
#确定batch_size
BATCH_SIZE = 16
#构建输入管道.读取时会将硬盘中的数据缓存,提高下一次读取的速度,加快训练
train_dataset = train_dataset.shuffle(buffer_size=train_count).batch(BATCH_SIZE).repeat()
test_dateset = test_dateset.batch(BATCH_SIZE).repeat()
# 建立模型
model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(64,(3,3),input_shape=(256,256,3),activation="relu"))
model.add(tf.keras.layers.Conv2D(64,(3,3),activation="relu"))
model.add(tf.keras.layers.MaxPool2D())
model.add(tf.keras.layers.Conv2D(128,(3,3),activation="relu"))
model.add(tf.keras.layers.Conv2D(128,(3,3),activation="relu"))
model.add(tf.keras.layers.MaxPool2D())
model.add(tf.keras.layers.Conv2D(256,(3,3),activation="relu"))
model.add(tf.keras.layers.Conv2D(256,(3,3),activation="relu"))
model.add(tf.keras.layers.MaxPool2D())
model.add(tf.keras.layers.Conv2D(512,(3,3),activation="relu"))
model.add(tf.keras.layers.Conv2D(512,(3,3),activation="relu"))
model.add(tf.keras.layers.MaxPool2D())
model.add(tf.keras.layers.Conv2D(1024,(3,3),activation="relu"))
model.add(tf.keras.layers.GlobalAveragePooling2D())
model.add(tf.keras.layers.Dense(1024,activation="relu"))
model.add(tf.keras.layers.Dense(256,activation="relu"))
model.add(tf.keras.layers.Dense(1,activation="sigmoid"))
model.summary()

model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["acc"])

steps_per_epoch = train_count//BATCH_SIZE
validation_steps = test_count//BATCH_SIZE
history = model.fit(train_dataset,epochs=5,steps_per_epoch=steps_per_epoch,validation_data=test_dateset,validation_steps=validation_steps)
plt.plot(history.epoch,history.history.get("acc"),label="acc")
plt.plot(history.epoch,history.history.get("val_acc"),label="val_acc")
plt.legend()#显示图例
plt.show()

plt.plot(history.epoch,history.history.get("loss"),label="loss")
plt.plot(history.epoch,history.history.get("val_loss"),label="val_loss")
plt.legend()#显示图例
plt.show()