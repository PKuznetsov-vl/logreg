import tensorflow as tf
from keras import regularizers
from keras.layers import Dense, Dropout, BatchNormalization, PReLU, Activation
from keras.models import Sequential
from keras.optimizer_experimental.sgd import SGD

from keras.regularizers import l2
from sklearn.model_selection import train_test_split
import numpy as np
#from mlxtend.plotting import plot_decision_regions
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
import tensorflow_datasets as tfds


#from __future__ import print_function
from keras.models import load_model
import keras
from keras.utils import np_utils
from sklearn.datasets import load_iris
#from sklearn import train_test_split
from keras.models import Sequential
from sklearn.linear_model import LogisticRegressionCV
from keras.layers.core import Activation
from keras.layers import Dense, Dropout, Activation
#from keras.optimizers import RMSprop
import numpy as np




def main():
    X, y = load_data('microchip_tests.txt')
    poly = PolynomialFeatures(degree=7)
    Xp = poly.fit_transform(X)
    # Split both independent and dependent variables in half for cross-validation
    train_X, test_X, train_y, test_y = train_test_split(Xp, y, train_size=0.5, random_state=0)
    # print(type(train_X),len(train_y),len(test_X),len(test_y))

    # lr = LogisticRegressionCV()
    # lr.fit(train_X, train_y)
    # pred_y = lr.predict(test_X)
    # print(pred_y)
    # print("Test fraction correct (LR-Accuracy) = {:.2f}".format(lr.score(test_X, test_y)))

    # def one_hot_encode_object_array(arr):
    #     uniques, ids = np.unique(arr, return_inverse=True)
    #     return np_utils.to_categorical(ids, len(uniques))
    #
    # # Dividing data into train and test data
    # train_y_ohe = one_hot_encode_object_array(train_y)
    # test_y_ohe = one_hot_encode_object_array(test_y)

    # Creating a model
    model = Sequential()
    model.add(Dense(1, input_dim=(Xp.shape[1]),kernel_initializer='random_normal',
                    kernel_regularizer=regularizers.L1(l1=0.01),
    bias_initializer='zeros'))
    model.add(Activation('sigmoid'))
    # model.add(Dense(1))
    # model.add(Activation('softmax'))

    # Compiling the model
    model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer='adam')

    # Actual modelling
    model.fit(train_X, train_y, verbose=0, batch_size=1, epochs=150)

    score, accuracy = model.evaluate(test_X, test_y, batch_size=16, verbose=0)

    # print("\n Test fraction correct (LR-Accuracy) logistic regression = {:.2f}".format(
    #     lr.score(test_X, test_y)))  # Accuracy is 0.83
    print("Test fraction correct (NN-Accuracy) keras  = {:.2f}".format(accuracy))  # Accuracy is 0.99
    print(model.predict(test_X))
    fig, ax = plot_decision_boundary(X=X, y=y, model=model, poly_featurizer=poly)
    fig.savefig("output.png")

def lr():
    X, y = load_data('ex2data1.txt')
    #print(X)
    poly = PolynomialFeatures(degree=7)
    xp = poly.fit_transform(X)
    C = 0.0001
    logit = LogisticRegression(C=C, n_jobs=-1, random_state=17)
    logit.fit(X, y)



    plt.scatter(X[y == 1, 0], X[y == 1, 1], c='green', label='1')
    plt.scatter(X[y == 0, 0], X[y == 0, 1], c='red', label='0')
    plt.xlabel("Тест 1")
    plt.ylabel("Тест 2")
    plt.title('2 класса. Регуляризация с C=0.0001')
    plt.legend()

    print("Доля правильных ответов классификатора на обучающей выборке:",
          round(logit.score(X, y), 3))

    print(logit.predict(X))

    plot_boundary(logit, X, y, grid_step=.01, poly_featurizer=poly)


def create_model(x):
    #x = np.array(x)
    #

    #Добавляем регуляризацию
    # создаем модель с бинарной классификацией
    model = keras.models.Sequential()
    # normalizer = layers.Normalization(input_shape=[1, ], axis=None)
    # normalizer.adapt(x)


    # 1 слой Выравнивает вход. Не влияет на размер партии.

    #model.add(tf.keras.layers.Flatten())
    #model.add(normalizer)

    #model.tf.keras.layers.experimental.preprocessing.Normalization.adapt(x)
    # 2 слой Dense реализует операцию: output = activation(dot(input, kernel) + bias), где активация
    # — это функция активации по элементам,
    # переданная в качестве аргумента активации, кернел — это матрица весов, созданная слоем,
    # а смещение — это вектор смещения, созданный слоем (применимо только в случае, если use_bias — True).

    #model.add(tf.keras.layers.Dense(x.shape[1], activation=tf.nn.relu, input_dim=x.shape[1]))
    #model.add(tf.keras.layers.Dense(10))
    model.add(keras.layers.Dense(1, activation='sigmoid', input_dim=x.shape[1], kernel_regularizer=regularizers.L1(l1=0.01),
                                   bias_initializer='zeros',kernel_initializer='random_normal'))
                                     #kernel_regularizer = regularizers.L1(l1=0.01),
                                    # bias_regularizer=regularizers.L2(1),
                                    # activity_regularizer=regularizers.L1(0.01)))
    # Компилируем модель оптимизатор= rmsprop
    # функция потерь бинарная энтропия




#еще способ
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


def binary_model():
    X, y = load_data('microchip_tests.txt')
    #полиномиальные признаки
    poly = PolynomialFeatures(degree=5)
    Xp = poly.fit_transform(X)

    #разбиваем датасет
    train_X, test_X, train_y, test_y = train_test_split(Xp, y, train_size=0.5, random_state=0)

    print(f"Количество строк в y_train по классам: {np.bincount(train_y)}")
    print(f"Количество строк в y_test по классам: {np.bincount(test_y)}")
    #Последовательная модель
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

    model.add(tf.keras.layers.Dense(Xp.shape[1], activation=tf.nn.relu, input_dim=Xp.shape[1]))
    model.add(tf.keras.layers.Dense(Xp.shape[1]*2,activation=tf.nn.relu))
    #model.add(tf.keras.layers.Dense(Xp.shape[1]*2)),
    model.add(
        keras.layers.Dense(1, activation='sigmoid', input_dim=Xp.shape[1]/2,
                           kernel_regularizer=regularizers.L1(l1=0.01),
                           activity_regularizer=regularizers.L1(0.01),
                           bias_initializer='zeros', kernel_initializer='random_normal'))
    #model.add(Dense(1,activation=tf.nn.relu))

    model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer='adam')
    model.fit(train_X, train_y, verbose=1, batch_size=1, epochs=100)
    #model.fit(train_X, train_Y, epochs=100, verbose=1)
    #выводим предсказания
    score, accuracy = model.evaluate(Xp, y, batch_size=16, verbose=0)

    # print("\n Test fraction correct (LR-Accuracy) logistic regression = {:.2f}".format(
    #     lr.score(test_X, test_y)))  # Accuracy is 0.83
    print("Test fraction correct (NN-Accuracy) keras  = {:.2f}".format(accuracy))  # Accuracy is 0.99


    #print("Предсказания")
    #print(model.predict(test_X).ravel())
    fig, ax = plot_decision_boundary(X=X, y=y, model=model,poly_featurizer=poly)
    fig.savefig("output.png")


def digits_class_low():
    x_train,x_test,y_train_new,y_test_new=getmnist()
    model=create_model(x_train)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )

    # тренируем модель  100 эпох
    model.fit(x_train, y_train_new, epochs=10, verbose=1,shuffle=True)
    #смотрим точность обучения
    print(model.metrics_names)
    print(model.evaluate(x_test,y_test_new))
    # выводим предсказания
    ans = model.predict(x_test)
    print("Предсказания")
    print(ans.ravel())
    model.save(r'./logisticRegressionKeras.hdf5')
    #fig, ax = plot_decision_boundary(X=x, y=y, model=model, poly_featurizer=poly)
    #fig.savefig("output.png")

if __name__ == '__main__':
    def   img():
        #digits_class_low()
        model = tf.keras.models.load_model(r'./logisticRegressionKeras.hdf5')
        im = convert_image('4.png')
        plt.imshow(im)
        plt.show()
        predict_input = im.reshape((-1, 784))
        prediction = model.predict(predict_input)
        print(prediction)
#img()
#lr()
binary_model()
#main()




