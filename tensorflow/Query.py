import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from sklearn.preprocessing import MinMaxScaler, scale
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from keras.models import Sequential
from keras.metrics import categorical_accuracy
from keras.callbacks import EarlyStopping
from keras.layers import LSTM, Dense, Activation, Dropout, Conv1D, MaxPooling1D, GlobalAveragePooling1D, MaxPool1D, \
    Flatten, Reshape
from keras.layers.embeddings import Embedding
from keras.layers import Bidirectional
from sklearn.linear_model import LinearRegression
from matplotlib.backends.backend_pdf import PdfPages
import keras



def load_data(file_name, sequence_length=10, split=0.80):
    x = np.array(
        pd.read_csv(r"E:\PyCharm Projects\TensorFlow\data\new.csv", error_bad_lines=False))  # data-RunManyTime
    df = pd.read_csv(file_name, sep=',')
    data_all = np.array(df).astype(float)
    scaler = MinMaxScaler()
    data_all = scaler.fit_transform(data_all)
    data = []
    for i in range(len(data_all)):
        data.append(data_all[i])
    reshaped_data = np.array(data).astype('float64')

    split_boundary = int(x.shape[0] * split)
    train_x = x[: split_boundary]
    test_x = x[split_boundary:]

    train_y = data_all[: split_boundary]
    test_y = data_all[split_boundary:]
    print(train_x.shape)
    return train_x, train_y, test_x, test_y, scaler


keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)



# 建模Dense
def build_model():
    # input_dim是输入的train_x的最后一个维度，train_x的维度为(n_samples, time_steps, input_dim)
    model = Sequential()
    model.add(Dense(200, input_shape=(10425,)))
    model.add(Activation('sigmoid'))


    model.add(Dense(200))
    model.add(Activation('sigmoid'))


    model.add(Dense(200))
    model.add(Activation('sigmoid'))

    # 输出结果
    model.add(Dense(3))
    # 最后一层用softmax
    model.add(Activation('sigmoid'))

    model.summary()
    model.compile(loss='mse',
                  optimizer='adam', metrics=['accuracy'])
    return model




def train_model(train_x, train_y, test_x, test_y):
    # print("model-train_x",train_x)
    # print("model-train_y",train_y)
    model = build_model()

    try:
        model.fit(train_x, train_y, batch_size=16, nb_epoch=400, validation_split=0.1)
        model.predict(train_x)
        predict = model.predict(test_x)
        predict = np.array(predict).astype(float)
        model.evaluate(test_x, test_y)
    except Exception as e:
        print(e)
    return predict, test_y


if __name__ == '__main__':
    train_x, train_y, test_x, test_y, scaler = load_data(
        r'E:\PyCharm Projects\TensorFlow\data\avgTime.csv')  # -RunManyTime
    predict_y, test_y = train_model(train_x, train_y, test_x, test_y)
