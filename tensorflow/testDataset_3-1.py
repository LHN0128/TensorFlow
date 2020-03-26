import tensorflow as tf
import numpy as np

# 把其中每一个元素变为一个组件
dataset = tf.data.Dataset.from_tensor_slices([1,2,3,4,5,6,7])
print(dataset)
#取出前四个数据，可以转换为一个可迭代对象iter
for ele in dataset.take(4):
    print(ele.numpy())

# tf.data.Dataset.from_tensor_slices中每个元素的属性大小（维数）必须相同
dataset_2 = tf.data.Dataset.from_tensor_slices([[1,2],[3,4],[5,6]])
for ele in dataset_2:
    print(ele.numpy())

#每一个元素都是一个tensor对象
dataset_dict = tf.data.Dataset.from_tensor_slices({"a":[1,2,3,4],"b":[6,7,8,9],"c":[12,13,14,15]})
for ele in dataset_dict:
    print(ele)

#对数据进行乱序处理,并且重复3次（数据量也扩大了3倍）
#batch(batch_size)确定一次取出多少个数据
dataset = dataset.shuffle(7)
dataset = dataset.repeat(count=3)
dataset = dataset.batch(3)
# dataset.map(tf.square)#用于调用一个函数对数据处理
dataset = dataset.map(tf.square)
for ele in dataset:
    print(ele.numpy())


