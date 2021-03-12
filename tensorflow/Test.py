# mnist ????demo

# import tensorflow as tf

import numpy as np
import matplotlib
import pandas as pd
import pylab
import matplotlib.pyplot as plt
from pylab import mpl
from sklearn import preprocessing
from tslearn.preprocessing import TimeSeriesResampler
from collections import defaultdict
import os
import time
import datetime
import shutil
import math
import sys
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from functools import reduce
import operator
from tkinter import _flatten


# 该函数可以平滑时间序列，让时间序列变得平滑
def smooth(x, window_len=100, window='hanning'):
    s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    # print(len(s))
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')

    y = np.convolve(w / w.sum(), s, mode='valid')
    return y


def read_tsdata(rootpath, str1, str2):
    pdata = []
    labelsave = {}
    label = []
    # 读取数据标签并存储
    for root, dirs, file in os.walk(rootpath):
        for files in file:
            if files.find(str2) >= 0:
                labelfile = open(rootpath + files)
                for line in labelfile:
                    labelstr = line.split(',')
                    labelsave[labelstr[0]] = labelstr[1].replace('\n', '')
                labelfile.close()

    # 读取数据,重采样至固定维数，并存储
    for root, dirs, file in os.walk(rootpath):
        for files in file:
            if files.find(str1) >= 0:
                print(rootpath + files)
                a = np.loadtxt(rootpath + files)
                x1 = a[:, 1]
                x1 = smooth(x1)
                _range = np.max(x1) - np.min(x1)
                x1 = (x1 - np.min(x1)) / _range
                x1 = TimeSeriesResampler(sz=300).fit_transform(x1)
                x1 = x1.ravel()

                ax = plt.gca()
                ax.invert_yaxis()
                plt.plot(x1)
                plt.show()
                pdata.append(x1)
                label.append(labelsave[files])

    # sam = reduce(operator.add, sam)
    return pdata, label


# 首先要根据数据集把数据给重采样了，转化为固定维度的数据
# 数据被转化为固定维度以后，还得去添加标签
# 把数据分为“数据本身”和“标签”两部分
# 放入keras训练学习
def produce_trainingdata(rootpath):
    tdata, tlabel = read_tsdata(rootpath, 'ref', 'label')
    lb = preprocessing.LabelBinarizer()  # 构建一个转换对象
    Y = lb.fit_transform(tlabel)  # 把标签转换为one_hot编码
    re_label = lb.inverse_transform(Y)

    return tdata, Y, tlabel  # 返回数据、标签


rootpath = r'E:\学习\太原理工大学\天文大数据竞赛\组长程序\traincollect\\'
data, label, name = produce_trainingdata(rootpath)
for i in range(len(data)):
    print(data[i], label[i], name[i])

import keras as K

# 定义模型
init = K.initializers.glorot_uniform(seed=1)
simple_adam = K.optimizers.Adam()
model = K.models.Sequential()
model.add(K.layers.Dense(units=30, input_dim=300, kernel_initializer=init, activation='relu'))
model.add(K.layers.Dense(units=600, kernel_initializer=init, activation='relu'))
model.add(K.layers.Dense(units=3, kernel_initializer=init, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer=simple_adam, metrics=['accuracy'])

# 训练模型
b_size = 2
max_epochs = 50
print("Starting training ")
h = model.fit(np.array(data), np.array(label), batch_size=b_size, epochs=max_epochs, shuffle=True, verbose=1)
print("Training finished \n")

# 随意写个时间序列测试一下
unknown = np.array([[0.1, 0.2, 0.5, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]], dtype=np.float32)
# unknown = np.array([[0.1,0.2,0.3,0.4,0.5,0.6,0.6,0.6,0.6,0.5,0.4,0.3,0.2,0.1,0.1]], dtype=np.float32)

# 重采样到300维，并把格式整理一下，以便能够输入
un = TimeSeriesResampler(sz=300).fit_transform(unknown)
un = un.ravel()
s = []
s.append(un)

# 预测
predicted = model.predict(np.array(s))
print(predicted)
















