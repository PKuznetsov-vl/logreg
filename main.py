import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
#from mlxtend.plotting import plot_decision_regions
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.model_selection import GridSearchCV
import matplotlib
from utils.dataset import getmnist, load_data
from utils.graphutils import plot_boundary, convert_image, plot_decision_boundary
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from tensorflow.keras import layers


def main():
    X1, y1 = load_data('ex2data1.txt')

    train_X, test_X, test_Y, train_Y = train_test_split(X1, y1, random_state=15, stratify=y1, test_size=0.5)
    print(train_Y)
    print(f"Количество строк в y_train по классам: {np.bincount(train_Y)}")
    print(f"Количество строк в y_test по классам: {np.bincount(test_Y)}")
    # train_X, train_Y, test_X, test_Y = load_data()

    num_features = train_X.shape[1]
    print(num_features)
    num_classes = train_Y.shape[0]
    print(num_classes)
    # nm = layers.Normalization(input_shape=[1 ], axis=None)
    # nm.adapt(num_features)
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=1, activation=tf.nn.sigmoid)# activation = 'sigmoid'
    ])

    model.compile(optimizer='sgd',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.fit(X1, y1, epochs=512)
    a = model.evaluate(test_X, test_Y)
    print(a)
    print(model.metrics_names)
    ans = model.predict(test_X)
    print(ans)



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
    #Добавляем регуляризацию
    # создаем модель с бинарной классификацией
    model = tf.keras.models.Sequential()
    # 1 слой Выравнивает вход. Не влияет на размер партии.
    model.add(tf.keras.layers.Flatten())
    # 2 слой Dense реализует операцию: output = activation(dot(input, kernel) + bias), где активация
    # — это функция активации по элементам,
    # переданная в качестве аргумента активации, кернел — это матрица весов, созданная слоем,
    # а смещение — это вектор смещения, созданный слоем (применимо только в случае, если use_bias — True).
    #model.add(tf.keras.layers.Dense(36, activation=tf.nn.relu, input_dim=x.shape[1]))
    model.add(tf.keras.layers.Dense(1, activation=tf.nn.sigmoid,input_dim=x.shape[1]))
                                    # kernel_regularizer=regularizers.L1L2(l1=1, l2=1),
                                    # bias_regularizer=regularizers.L2(1),
                                    # activity_regularizer=regularizers.L2(1)))
    # Компилируем модель оптимизатор= rmsprop
    # функция потерь бинарная энтропия

    return model


def binary_model():
    X, y = load_data('ex2data1.txt')
    #полиномиальные признаки
    poly = PolynomialFeatures(degree=9)
    xp = poly.fit_transform(X)

    print(xp[0])
    #разбиваем датасет
    train_X, test_X, test_Y, train_Y = train_test_split(xp, y, random_state=15, stratify=y, test_size=0.5)
    print(train_Y)
    print(f"Количество строк в y_train по классам: {np.bincount(train_Y)}")
    print(f"Количество строк в y_test по классам: {np.bincount(test_Y)}")

    model=create_model(train_X)
    model.compile(optimizer='rmsprop',loss='binary_crossentropy' ,metrics=['binary_accuracy'])#loss='binary_crossentropy'
    # тренируем модель  100 эпох
    model.fit(train_X, train_Y, epochs=100, verbose=1)
    #выводим предсказания
    ans = model.predict(test_X)
    print("Предсказания")
    print(ans.ravel())
    fig, ax = plot_decision_boundary(X=X, y=y, model=model,poly_featurizer=poly)
    fig.savefig("output.png")









def digits_class_low():
    x_train,x_test,y_train_new,y_test_new=getmnist()
    model=create_model(x_train)
    model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['binary_accuracy'])
    # тренируем модель  100 эпох
    model.fit(x_train, y_train_new, epochs=5, verbose=1,shuffle=True)
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
    digits_class_low()
    model = tf.keras.models.load_model(r'./logisticRegressionKeras.hdf5')
    im = convert_image('0.png')
    plt.imshow(im)
    plt.show()
    #lr()
    #binary_model()
    predict_input = im.reshape((-1, 784))
    prediction = model.predict(predict_input)
    print(prediction)




