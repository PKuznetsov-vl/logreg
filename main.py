import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from numpy import linspace, meshgrid, c_
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
from mlxtend.plotting import plot_decision_regions
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.model_selection import GridSearchCV





def plot_model_out(x,y,model):
  """
  x,y: 2D MeshGrid input
  model: Keras Model API Object
  """
  grid = np.stack((x,y))
  grid = grid.T.reshape(-1,2)
  outs = model.predict(grid)
  y1 = outs.T[0].reshape(x.shape[0],x.shape[0])
  plt.contourf(x,y,y1)
  plt.show()

def load_data(path):
    data = pd.read_csv(path,
                       header=None, names=('test1', 'test2', 'released'))
    # информация о наборе данных
    data.info()
    X = data.iloc[:, :2].values
    print(X)
    y = data.iloc[:, 2].values
    print(y)

    plt.scatter(X[y == 1, 0], X[y == 1, 1], c='green', label='1')
    plt.scatter(X[y == 0, 0], X[y == 0, 1], c='red', label='0')
    plt.xlabel("Тест 1")
    plt.ylabel("Тест 2")
    plt.title('Ex2data1 нет нормализации')
    plt.legend()
    plt.plot()
    #plt.show()
    return X,y


def plot_boundary(clf, X, y, grid_step=.01, poly_featurizer=None):
    x_min, x_max = X[:, 0].min() - .1, X[:, 0].max() + .1
    y_min, y_max = X[:, 1].min() - .1, X[:, 1].max() + .1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, grid_step),
                         np.arange(y_min, y_max, grid_step))

    # каждой точке в сетке [x_min, m_max]x[y_min, y_max]
    # ставим в соответствие свой цвет
    Z = clf.predict(poly_featurizer.transform(np.c_[xx.ravel(), yy.ravel()]))
    Z = Z.reshape(xx.shape)
    plt.contour(xx, yy, Z, cmap=plt.cm.Paired)


def plot_decision_boundary(X, y, model, steps=1000, cmap='Paired'):
    """
    Function to plot the decision boundary and data points of a model.
    Data points are colored based on their actual label.
    """
    cmap = get_cmap(cmap)

    # Define region of interest by data limits
    xmin, xmax = X[:,0].min() - 1, X[:,0].max() + 1
    ymin, ymax = X[:,1].min() - 1, X[:,1].max() + 1
    steps = 1000
    x_span = linspace(xmin, xmax, steps)
    y_span = linspace(ymin, ymax, steps)
    xx, yy = meshgrid(x_span, y_span)

    # Make predictions across region of interest
    labels = model.predict(c_[xx.ravel(), yy.ravel()])

    # Plot decision boundary in region of interest
    z = labels.reshape(xx.shape)

    fig, ax = plt.subplots()
    train_labels = model.predict(X)
    ax.contourf(xx, yy, z, cmap=cmap, alpha=0.5)

    # Get predicted labels on training data and plot
    train_labels = model.predict(X)
    ax.scatter(X[:,0], X[:,1], c=y.ravel(), cmap=cmap, lw=0)

    return fig, ax






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
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    model.compile(optimizer='sgd',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(X1, y1, epochs=10)
    a = model.evaluate(test_X, test_Y)
    print(a)
    print(model.metrics_names)
    ans = model.predict(test_X).flatten()
    # plt.plot(test_X, ans, color='k', label='Predictions')

    # x = tf.linspace(0.0, 250, 251)
    print(ans)
    print(test_X)

    # plt.scatter(X1, y1, label='Data')
    plt.plot(test_X[:1], ans, color='k', label='Predictions')
    plt.xlabel('Horsepower')
    plt.ylabel('MPG')
    plt.legend()


X, y = load_data('ex2data1.txt')
print(X)

C = 1e-2
logit = LogisticRegression(C=C, n_jobs=-1, random_state=17)
logit.fit(X, y)

#plot_boundary(logit, X, y, grid_step=.01, poly_featurizer=poly)

# plt.scatter(X[y == 1, 0], X[y == 1, 1], c='green', label='Выпущен')
# plt.scatter(X[y == 0, 0], X[y == 0, 1], c='red', label='Бракован')
# plt.xlabel("Тест 1")
# plt.ylabel("Тест 2")
# plt.title('2 теста микрочипов. Логит с C=0.01')
# plt.legend();

print("Доля правильных ответов классификатора на обучающей выборке:",
round(logit.score(X, y), 3))

print(logit.predict(X))