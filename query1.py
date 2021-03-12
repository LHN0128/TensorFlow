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


def c2i(c):
    data = []
    for i in range(len(c)):
        data.append(int(c[i]))
    return data


def f_a2(file):
    fin = open(file)
    a2 = []
    lines = fin.readlines()
    for line in lines:
        line = line.strip("\n\'").split(",")
        # print(line)
        a2.append(c2i(line))

    return a2


def changex(temp, position):
    return int(temp * 10)


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


def build_model():
    # input_dim是输入的train_x的最后一个维度，train_x的维度为(n_samples, time_steps, input_dim)
    model = Sequential()
    model.add(Dense(200, input_shape=(10425,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.8))

    model.add(Dense(200))
    model.add(Activation('relu'))
    model.add(Dropout(0.8))

    # 输出结果
    model.add(Dense(3))
    # 最后一层用softmax
    model.add(Activation('relu'))

    model.summary()

    model.compile(loss="mse",
                  optimizer='adam', metrics=['accuracy'])
    return model




# 准确率与损失函数图
def training_plot(history):
    print("history.history.keys=========", history.history.keys())
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    acc = history.history['acc']
    val_acc = history.history['val_acc']

    # make a figure
    plt.figure()
    plt.plot(loss, label='训练集')
    plt.plot(val_loss, label='有效集')
    plt.xlabel('迭代次数')
    plt.ylabel('损失')
    # plt.title('Loss on Training and Validation Data')
    plt.legend()
    # plt.savefig('E:\\DATA\\sqlResult\\data-RunManyTime\\png\\png\\1-2728-loss-256.png')  # data--RunManyTime
    plt.show()
    # plt.gca().xaxis.set_major_formatter(FuncFormatter(changex))
    plt.close()

    # subplot acc
    plt.figure()
    plt.plot(acc, label='train_acc')
    plt.plot(val_acc, label='test_acc')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy  on Training and Validation Data')
    plt.legend()
    # plt.savefig('E:\\DATA\\sqlResult\\data-RunManyTime\\png\\png\\1-2728-acc-256.png')  # -RunManyTime
    plt.show()
    # plt.gca().xaxis.set_major_formatter(FuncFormatter(changex))
    plt.close()


def train_model(train_x, train_y, test_x, test_y):
    # print("model-train_x",train_x)
    # print("model-train_y",train_y)
    model = build_model()

    try:
        history = model.fit(train_x, train_y, batch_size=256, nb_epoch=1000, validation_split=0.1)
        training_plot(history)
        predict_train = model.predict(train_x)
        predict_train = np.array(predict_train).astype(float)
        predict = model.predict(test_x)
        predict = np.array(predict).astype(float)
        # predict = np.array(predict).astype(float))
        # print("mean_relative_error = ",mean_relative_error(test_y, predict_y,))
        print("mean_squared_error = ", mean_squared_error(test_y, predict))
        print("mean_absolute_error = ", mean_absolute_error(test_y, predict))
        print("r2_score = ", r2_score(test_y, predict))
    except KeyboardInterrupt:
        print('1')
        print('1')
    print('2')
    print('2')
    try:
        List1 = [x[0] for x in test_y]
        list1 = [x[0] for x in predict]
        list2 = [x[1] for x in predict]
        list3 = [x[2] for x in predict]

        list11 = [x[0] for x in test_y]
        list22 = [x[1] for x in test_y]
        list33 = [x[2] for x in test_y]

        loss, accuracy = model.evaluate(test_x, test_y)

        print('loss:', loss)
        print('accuracy:', accuracy)
        '''
        print("============================================================================ ")
        print("========================================List1==================================== ",List1)
        print("============================================================================ ")
        print("============================================================================ ")
        print("========================================list1==================================== ",list1)
        '''
        print("============================================================================ ")
        print("firstSql:mean_squared_error = ", mean_squared_error(list1, list11))
        print("firstSql:mean_absolute_error = ", mean_absolute_error(list1, list11))
        print("firstSql:r2_score = ", r2_score(list1, list11))

        # print("mean_relative_error = ",mean_relative_error(list2, list22,))
        print("secondSql:mean_squared_error = ", mean_squared_error(list2, list22))
        print("secondSql:mean_absolute_error = ", mean_absolute_error(list2, list22))
        print("secondSql:r2_score = ", r2_score(list2, list22))

        # print("mean_relative_error = ",mean_relative_error(list3, list33,))
        print("thirdSql:mean_squared_error = ", mean_squared_error(list3, list33))
        print("thirdSql:mean_absolute_error = ", mean_absolute_error(list3, list33))
        print("thirdSql:r2_score = ", r2_score(list3, list33))
        print("============================================================================ ")
        fig = plt.figure(figsize=(12, 12))
        ax1 = fig.add_subplot(311)
        ax1.plot(list1, label='predict', color='red', linestyle=':')
        ax1.plot(list11, label='true', color='green', linestyle='-')
        ax1.legend(loc='upper left')
        ax1.set_xlabel('Test Count')
        ax1.set_ylabel('Time')
        ax1.set_title('Predicted and real data of the first query in query combination')

        fig = plt.figure(figsize=(12, 12))
        ax1 = fig.add_subplot(311)
        ax1.plot(list2, label='predict', color='red', linestyle=':')
        ax1.plot(list22, label='true', color='green', linestyle='-')
        ax1.legend(loc='upper left')
        ax1.set_xlabel('Test Count')
        ax1.set_ylabel('Time')
        ax1.set_title('Predicted and real data of the second query in query combination')

        fig = plt.figure(figsize=(12, 12))
        ax1 = fig.add_subplot(311)
        ax1.plot(list3, label='predict', color='red', linestyle=':')
        ax1.plot(list33, label='true', color='green', linestyle='-')
        ax1.legend(loc='upper left')
        ax1.set_xlabel('Test Count')
        ax1.set_ylabel('Time')
        ax1.set_title('Predicted and real data of the third query in query combination')

        fig = plt.figure(figsize=(12, 12))
        ax1 = fig.add_subplot(411)
        ax1.plot(predict, label='predict', color='red', linestyle=':')
        ax1.plot(test_y, label='true', color='green', linestyle='-')
        ax1.legend(loc='upper left')
        ax1.set_xlabel('Test Count')
        ax1.set_ylabel('Time')
        ax1.set_title('Predict Data and True Data')



    except Exception as e:
        print(e)
    return predict, test_y


if __name__ == '__main__':
    train_x, train_y, test_x, test_y, scaler = load_data(
        r'E:\PyCharm Projects\TensorFlow\data\avgTime.csv')  # -RunManyTime
    print(train_x.shape)
    print(train_y.shape)
    print(test_x.shape)
    print(test_y.shape)
    # train_x = np.reshape(train_x,(train_x.shape[0],train_x.shape[1],10425))
    # test_x = np.reshape(test_x,(test_x.shape[0],test_x.shape[1],10425))
    predict_y, test_y = train_model(train_x, train_y, test_x, test_y)
    # predict_y = scaler.inverse_transform([[i] for i in predict_y])
    # test_y = scaler.inverse_transform(test_y)
    # make a figure
    test_y = scaler.inverse_transform(test_y)
    predict_y = scaler.inverse_transform(predict_y)
    # print("mean_relative_error = ",mean_relative_error(test_y, predict_y,))
    print("mean_squared_error = ", mean_squared_error(test_y, predict_y))
    print("mean_absolute_error = ", mean_absolute_error(test_y, predict_y))
    print("r2_score = ", r2_score(test_y, predict_y))
    print("============================================================================ ")

    list1 = [x[0] for x in predict_y]
    list2 = [x[1] for x in predict_y]
    list3 = [x[2] for x in predict_y]

    list11 = [x[0] for x in test_y]
    list22 = [x[1] for x in test_y]
    list33 = [x[2] for x in test_y]

    # print("mean_relative_error = ",mean_relative_error(list1, list11,))
    print("firstSql:mean_squared_error = ", mean_squared_error(list1, list11))
    print("firstSql:mean_absolute_error = ", mean_absolute_error(list1, list11))
    print("firstSql:r2_score = ", r2_score(list1, list11))

    # print("mean_relative_error = ",mean_relative_error(list2, list22,))
    print("secondSql:mean_squared_error = ", mean_squared_error(list2, list22))
    print("secondSql:mean_absolute_error = ", mean_absolute_error(list2, list22))
    print("secondSql:r2_score = ", r2_score(list2, list22))

    # print("mean_relative_error = ",mean_relative_error(list3, list33,))
    print("thirdSql:mean_squared_error = ", mean_squared_error(list3, list33))
    print("thirdSql:mean_absolute_error = ", mean_absolute_error(list3, list33))
    print("thirdSql:r2_score = ", r2_score(list3, list33))
    plt.figure(figsize=(16, 5))
    plt.plot(list1, label='predict', color='green', linestyle='-')
    plt.plot(list11, label='true', color='red', linestyle=':')
    plt.legend(loc='upper left')
    plt.xlabel('预测组合数')
    plt.ylabel('时间/s')
    # plt.title('Predicted and real data of the first query in query combination')
    # 改变坐标轴刻度
    # plt.gca().xaxis.set_major_formatter(FuncFormatter(changex))
    # plt.savefig('E:\\DATA\\sqlResult\\data-RunManyTime\\png\\png\\1-2728-1-256.png')  # -RunManyTime
    plt.show()
    plt.close()

    plt.figure(figsize=(16, 5))
    plt.plot(list2, label='predict', color='green', linestyle='-')
    plt.plot(list22, label='true', color='red', linestyle=':')
    plt.legend(loc='upper left')
    plt.xlabel('Predict Count')
    plt.ylabel('Time/s')
    # plt.title('Predicted and real data of the first query in query combination')
    # 改变坐标轴刻度
    # plt.gca().xaxis.set_major_formatter(FuncFormatter(changex))
    # plt.savefig('E:\\DATA\\sqlResult\\data-RunManyTime\\png\\png\\1-2728-2-256.png')  # -RunManyTime
    plt.show()
    plt.close()

    plt.figure(figsize=(16, 5))
    plt.plot(list3, label='predict', color='green', linestyle='-')
    plt.plot(list33, label='true', color='red', linestyle=':')
    plt.legend(loc='upper left')
    plt.xlabel('Predict Count')
    plt.ylabel('Time/s')
    # plt.title('Predicted and real data of the first query in query combination')
    # 改变坐标轴刻度
    # plt.gca().xaxis.set_major_formatter(FuncFormatter(changex))
    # plt.savefig('E:\\DATA\\sqlResult\\data-RunManyTime\\png\\png\\1-2728-3-256.png')  # data-RunManyTime
    plt.show()
    plt.close()

    plt.figure(figsize=(16, 5))
    plt.plot(predict_y, label='predict', color='green', linestyle='-')
    plt.plot(test_y, label='true', color='red', linestyle=':')
    plt.legend(loc='upper left')
    plt.xlabel('Predict Count')
    plt.ylabel('Time/s')
    # plt.title('Predicted and real data of the first query in query combination')
    # 改变坐标轴刻度
    # plt.gca().xaxis.set_major_formatter(FuncFormatter(changex))
    # plt.savefig('E:\\DATA\\sqlResult\\data-RunManyTime\\png\\png\\1-2728-all-256.png')  # -RunManyTime
    plt.show()
    plt.close()
