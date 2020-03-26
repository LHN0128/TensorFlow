import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#读取是数据集
data = pd.read_csv(r'E:\PyCharm Projects\Test\data\Advertising.csv')
print(data.head())
#绘制TV和sales之间的关系。通过绘制不同的图来发现不同变量对sales的关系。
plt.scatter(data.TV,data.sales)
#plt.scatter(data.newspaper,data.sales)
plt.show()


# 建立模型，已知TV，radio，newspaper上广告的投放量时，预测销量
#iloc[行，列]，取所有的行，用：，取除去第一列和最后一列是1：-1
x = data.iloc[:,1:-1]
y = data.iloc[:,-1]
#中间层10个单元的隐藏层，3个输入加一个偏置共4个输入，10个单元，因此权重矩阵是4*10=40个params
model = tf.keras.Sequential([tf.keras.layers.Dense(10,input_shape=(3,),activation='relu'),
                             tf.keras.layers.Dense(1)
  ])
print(model.summary())
#编译模型，添加优化方法（其中包括学习率）和损失函数
model.compile(optimizer='adam',loss='mse')
#训练模型
model.fit(x,y,epochs=100)

#预测数据：反过来预测一下数据集中前十个
test = data.iloc[:10,1:-1]

print(model.predict(test))