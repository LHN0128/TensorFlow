import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#读取数据集,如果第一行就是数据而没有表头，那么就添加header=None参数
data = pd.read_csv(r'E:\PyCharm Projects\TensorFlow\data\credit-a.csv',header=None)
print(data.head())
#查看是或不是的分布情况
print(data.iloc[:, -1].value_counts())
# 提取输入x和输出y。这里把-1换成0来表示负面数据
x = data.iloc[:,:-1]
y = data.iloc[:,-1].replace(-1,0)
print(x.shape)
print(y.shape)
print(y)
model = tf.keras.Sequential()
#第一层要告诉输入数据的形状，input_shape，15个数据输入,后面每一个隐藏层都是10个结点
model.add(tf.keras.layers.Dense(10,input_shape=(15,),activation='relu'))
model.add(tf.keras.layers.Dense(10,activation='relu'))
model.add(tf.keras.layers.Dense(10,activation='relu'))
model.add(tf.keras.layers.Dense(10,activation='relu'))
#逻辑回归最后一层输出层要使用sigmoid激活，判断是非
model.add(tf.keras.layers.Dense(1,activation='sigmoid'))
#二元分类使用binary_crossentropy交叉熵，metrics用来计算每一个epoch的精度
model.compile(optimizer=tf.keras.optimizers.Adam(0.0005),loss="binary_crossentropy",metrics=["acc"])

history = model.fit(x,y,epochs=1000)
#history里有history参数，history.history是一个字典，然后查看里面的参数：["loss","acc"]
print(history.history.keys())
#绘制线段图使用plt.plot(history.epoch,history.history.get("loss"))
# history.epoch是表示次数作为x轴，loss或acc作为y轴，看一下损失的下降曲线

#plt.plot(history.epoch,history.history.get('loss'))
#plt.show()

plt.plot(history.epoch,history.history.get('loss'))
plt.show()

test1 = data.iloc[:5,:-1]
print(model.predict(test1))

test2 = data.iloc[648:,:-1]
print(model.predict(test2))
# 保存模型
model.save("./logisticRegression.h5")
# newmodel = tf.keras.models.load_model("./logisticRegression.h5")