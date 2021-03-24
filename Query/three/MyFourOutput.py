#该文件输入未知的查询组合执行计划特征FEP，输出查询交互特征FQI
#输出的是经过归一化后参数的交互值
import tensorflow as tf
import numpy as np
import pandas as pd

#加载模型
model = tf.keras.models.load_model(r"E:\PyCharm_Projects\TensorFlow\Query\models\MyFourQueryModel.h5")
#预测的特征输入
predict = pd.read_csv(r'E:\学习\太原理工大学\课题\test\four\FourFeature.csv', header=None,
                          low_memory=False)
predict = predict.iloc[:, :12300]

predict_list = [predict]
predict_list = np.concatenate(predict_list, axis=0)
predict_list = predict_list.reshape(1,2094,12300)
model.summary()
#预测之后的labels
predict_list = model.predict(predict_list)
#写入文件
np.savetxt(r'E:\学习\太原理工大学\课题\test\four\FourFQI.csv',predict_list[0],delimiter=',')