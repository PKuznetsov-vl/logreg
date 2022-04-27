import pandas as pd
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import np_utils
from sklearn import datasets, metrics,preprocessing

def getmnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # print("old",x_train.shape)
    print("oldy",y_train.shape[0])
    print(y_train[0:10])
   # y_test = np_utils.to_categorical(y_test)
    #num_classes = y_train.shape[0]
    #print(num_classes)
    # Генератор списков
    x_train_new, y_train_new = x_train[(y_train == 0) | (y_train == 1)], y_train[(y_train == 0) | (y_train == 1)]
    # изменяем массив
    x_train_final = x_train_new.reshape((-1, 784))
    print('final', x_train_final.shape)

    x_test_new, y_test_new = x_test[(y_test == 0) | (y_test == 1)], y_test[(y_test == 0) | (y_test == 1)]
    x_test_final = x_test_new.reshape((-1, 784))
    # Нормализируем
    x_train_final = x_train_final / 255
    x_test_final = x_test_final / 255
    return x_train_final, x_test_final, y_train_new, y_test_new


def load_data(path):
    data = pd.read_csv(path,
                       header=None, names=('test1', 'test2', 'released'))
    # информация о наборе данных
    data.info()
    X = data.iloc[:, :2].values
    #print(X)
    y = data.iloc[:, 2].values
    #print(y)
    #scaler = preprocessing.StandardScaler()
    # print data['age'].shape,type(data['age'])
    # print data['age'].reshape(-1,1).shape, type(data['age'].reshape(-1,1))

    # Конвертировать серию в numy (100, 1)

    plt.scatter(X[y == 1, 0], X[y == 1, 1], c='red', label='1')
    plt.scatter(X[y == 0, 0], X[y == 0, 1], c='blue', label='0')
    plt.xlabel("Тест 1")
    plt.ylabel("Тест 2")
    plt.title('Ex2data1 нет нормализации')
    plt.legend()
    plt.plot()
    #X = scaler.fit_transform(X)
    #print(X)
    #plt.show()
    return X,y