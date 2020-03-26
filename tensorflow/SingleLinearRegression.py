import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
#读取数据集，然后绘图，字符串前加r是对全部的\转义

data = pd.read_csv(r'E:\PyCharm Projects\Test\data\Income.csv')
plt.scatter(data.Education,data.Income)
plt.show()

#顺序模型：一个输入，一个输出，一层一层的搭建的模型，就叫sequencial
x = data.Education
y = data.Income
#创建sequential模型
model = tf.keras.Sequential()
#Dense层是最常用的层，Dense(该层神经单元数，input_shape=(变量数，))
model.add(tf.keras.layers.Dense(1,input_shape=(1,)))
print(model.summary())
#model.summary()显示模型信息
# _________________________________________________________________
# Layer (type)                 Output Shape                       Param #
# =================================================================
# dense (Dense)                (None, 1) 样本数，输出单元数            2    参数2个（y=ax+b）
# none表示样本数目


#内置了很多优化方法，最常用的是adam，这里设置MSE损失函数。adam优化算法的默认学习速率是0.001
model.compile(optimizer='adam',loss='mse')

#epochs表示对所有的数据进行训练的次数。开始训练一个模型。至此，模型训练结束
history = model.fit(x,y,epochs=1000)
# print(history)
#因为输入的时候的格式是pandas.Series，因此预测的时候也应该是一个series的格式
model.predict(pd.Series([20]))
