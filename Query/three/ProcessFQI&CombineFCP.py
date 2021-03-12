#该文件处理FQI为onehot编码，并合并FCP
import tensorflow as tf
import numpy as np
import pandas as pd
import os

pd.set_option('display.max_columns', 1000) #显示完整的列
pd.set_option('display.max_rows', 1000) #显示完整的行
pd.set_option('display.width', 1000) #显示最大的行宽
pd.set_option('display.max_colwidth', 1000) #显示最大的列宽



#读取查询交互特征FQI并作处理
FQI = pd.read_csv(r'E:\学习\太原理工大学\课题\test\three\ThreeFQI.csv', header=None,
                          low_memory=False)
FQI = FQI*1000
FQI = np.floor(FQI)#全部向下取整，表示在区间内

# #这里将每一个数都进行了one-hot编码。例如2090*12的数据，最大为635，转换后为2090*12*635
FQI = tf.keras.utils.to_categorical(FQI)
print(FQI.shape)

FCP = pd.read_csv(r'E:\学习\太原理工大学\课题\test\three\ThreeFCP.csv', header=None,
                          low_memory=False)
FCP = FCP.iloc[:,:9225]
FCP = np.array(FCP)
FCPFQI = []
FCPFQIone = []
#遍历FQI中每一个值
for i in range(2090-1):
    # print(np.hstack(FQI[i]).shape)
    # print(FCP[i].shape)
    #这是拼好的一行np.hstack((a,b))
    FCPFQIone = np.hstack((np.hstack(FQI[i]), FCP[i]))
    if i==0:
        FCPFQI = FCPFQIone
    # 向一个空的矩阵中不断追加合并后的行np.vstack((array,row))
    FCPFQI = np.vstack((FCPFQI, FCPFQIone))



print(FCPFQI.shape)
#写入csv
dataFrame = pd.DataFrame(FCPFQI)
dataFrame.to_csv(r"E:\学习\太原理工大学\课题\test\three\threeFCPFQI.csv",index=False,index_label=False,header=False)