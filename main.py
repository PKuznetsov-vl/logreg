import tensorflow as tf
from keras import regularizers
from keras.layers import Dense, Dropout, BatchNormalization, PReLU, Activation
from keras.models import Sequential
from keras.optimizer_experimental.sgd import SGD

from keras.regularizers import l2
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

# from mlxtend.plotting import plot_decision_regions
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.model_selection import GridSearchCV
import matplotlib
from utils.dataset import getmnist, load_data
from utils.graphutils import plot_boundary, convert_image, plot_decision_boundary

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from tensorflow.keras import layers
#import tensorflow_datasets as tfds

# from __future__ import print_function
from keras.models import load_model
import keras
from keras.utils import np_utils
from sklearn.datasets import load_iris
# from sklearn import train_test_split
from keras.models import Sequential
from sklearn.linear_model import LogisticRegressionCV
from keras.layers.core import Activation
from keras.layers import Dense, Dropout, Activation
# from keras.optimizers import RMSprop
import numpy as np


def first_model(path,poly=None):

    X, y = load_data('ex2data1.txt','Ex2data1 нет нормализации')
    if poly:
        # полиномиальные признаки
        #poly = PolynomialFeatures(degree=3)
        X = poly.fit_transform(X)

    # разбиваем датасет
    train_X, test_X, train_y, test_y = train_test_split(X, y, train_size=0.5, random_state=1)
    # Creating a model
    model = Sequential()
    # model.add(Dense(1))
    model.add(tf.keras.layers.Dense(16, activation=tf.nn.relu, input_dim=X.shape[1]))

    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
    # model.add(tf.keras.layers.Dropout(0.6))
    model.add(
        keras.layers.Dense(1, activation='sigmoid', input_dim=4, kernel_regularizer=regularizers.L1(l1=0.0000001),
                           bias_initializer='zeros', kernel_initializer='random_normal'))

    # Compiling the model
    model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer='adam')

    # Actual modelling
    model.fit(train_X, train_y, verbose=1,batch_size=1, epochs=180)

    score, accuracy = model.evaluate(X, y, batch_size=16, verbose=0)


    print("Test fraction correct (NN-Accuracy) keras  = {:.2f}".format(accuracy))  # Accuracy is 0.99
    # print(model.predict(test_X))

    if poly:
        fig, ax = plot_decision_boundary(X=X, y=y, model=model, poly_featurizer=poly)
    else:
        fig, ax = plot_decision_boundary(X=X, y=y, model=model, poly_featurizer=None)
    fig.savefig(path)
    plot_boundary(model, X, y, grid_step=.01, poly_featurizer=None)

def microchips_lr():
    X, y = load_data('ex2data1.txt','Ex2data2 нет нормализации')
    # print(X)
    poly = PolynomialFeatures(degree=3)
    Xp = poly.fit_transform(X)
    C = 0.0001
    logit = LogisticRegression(C=C, n_jobs=-1, random_state=17)
    logit.fit(Xp, y)

    plt.scatter(X[y == 1, 0], X[y == 1, 1], c='red', label='1')
    plt.scatter(X[y == 0, 0], X[y == 0, 1], c='blue', label='0')
    plt.xlabel("Тест 1")
    plt.ylabel("Тест 2")
    plt.title('2 класса. Регуляризация с C=0.0001 ')
    plt.legend()

    print("Доля правильных ответов классификатора на обучающей выборке:",
          round(logit.score(Xp, y), 3))

    print(logit.predict(Xp))

    plot_boundary(logit, X, y, grid_step=.01, poly_featurizer=poly)


def create_model(X):
    # x = np.array(x)
    #

    # Добавляем регуляризацию
    # создаем модель с бинарной классификацией
    # Последовательная модель
    model = keras.models.Sequential()
    # normalizer = layers.Normalization(input_shape=[1, ], axis=None)
    # normalizer.adapt(x)

    # 1 слой Выравнивает вход. Не влияет на размер партии.

    model.add(tf.keras.layers.Flatten())
    # model.add(normalizer)

    # model.tf.keras.layers.experimental.preprocessing.Normalization.adapt(x)
    # 2 слой Dense реализует операцию: output = activation(dot(input, kernel) + bias), где активация
    # — это функция активации по элементам,
    # переданная в качестве аргумента активации, кернел — это матрица весов, созданная слоем,
    # а смещение — это вектор смещения, созданный слоем (применимо только в случае, если use_bias — True).

    model.add(tf.keras.layers.Dense(X.shape[1], activation=tf.nn.relu, input_dim=X.shape[1]))
    model.add(tf.keras.layers.Dense(X.shape[1] * 2, activation=tf.nn.relu))
    # model.add(tf.keras.layers.Dense(Xp.shape[1]*2)),
    model.add(
        keras.layers.Dense(1, activation='sigmoid', input_dim=X.shape[1] / 2,
                           kernel_regularizer=regularizers.L1(l1=0.01),
                           activity_regularizer=regularizers.L1(0.01),
                           bias_initializer='zeros', kernel_initializer='random_normal'))

    # еще способ
    # layer = tf.keras.layers.experimental.preprocessing.Normalization()
    # layer.adapt(train_X)
    #
    # model = tf.keras.Sequential(
    #     [
    #         layer,
    #         tf.keras.layers.Dense(64, activation=tf.nn.relu, input_dim=xp.shape[1]),
    #         tf.keras.layers.Dense(1, activation=tf.nn.sigmoid),
    #     ]
    # )
    return model


def microchops_nn():
    X, y = load_data('microchip_tests.txt','tst')
    # полиномиальные признаки
    poly = PolynomialFeatures(degree=7)
    Xp = poly.fit_transform(X)

    # разбиваем датасет
    train_X, test_X, train_y, test_y = train_test_split(Xp, y, train_size=0.5, random_state=0)

    print(f"Количество строк в y_train по классам: {np.bincount(train_y)}")
    print(f"Количество строк в y_test по классам: {np.bincount(test_y)}")

    model=create_model(Xp)

    model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer='adam')
    model.fit(train_X, train_y, verbose=1, batch_size=1, epochs=100)
    # model.fit(train_X, train_Y, epochs=100, verbose=1)
    # выводим предсказания
    score, accuracy = model.evaluate(Xp, y, batch_size=16, verbose=0)

    # print("\n Test fraction correct (LR-Accuracy) logistic regression = {:.2f}".format(
    #     lr.score(test_X, test_y)))  # Accuracy is 0.83
    print("Test fraction correct (NN-Accuracy) keras  = {:.2f}".format(accuracy))  # Accuracy is 0.99

    # print("Предсказания")
    # print(model.predict(test_X).ravel())
    fig, ax = plot_decision_boundary(X=X, y=y, model=model, poly_featurizer=poly)
    fig.savefig("micro.png")




if __name__ == '__main__':
    #сравинить с ski-kit
        # microchips_lr()
        # print('работа модели на датсете ex2data1.txt')
        #first_model(path='1.png')
        # print('работа модели на датсете ex2data1.txt c полиномиальными признаками 3ей степени')
        # poly = PolynomialFeatures(degree=2)
        # first_model(poly=poly,path='2.png')
        print('работа модели на датсете ex2data2.txt c полиномиальными признаками 9ой степени')
        microchops_nn()
