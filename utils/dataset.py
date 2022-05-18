import pandas as pd
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import np_utils
from sklearn import datasets, metrics,preprocessing



def load_data(path,title):
    data = pd.read_csv(path,
                       header=None, names=('test1', 'test2', 'released'))
    # информация о наборе данных
    data.info()
    X = data.iloc[:, :2].values
    #print(X)
    y = data.iloc[:, 2].values
    plt.scatter(X[y == 1, 0], X[y == 1, 1], c='red', label='1')
    plt.scatter(X[y == 0, 0], X[y == 0, 1], c='blue', label='0')
    plt.xlabel("Тест 1")
    plt.ylabel("Тест 2")
    plt.title(title)#
    plt.legend()
    plt.plot()
    #plt.show()
    return X,y
