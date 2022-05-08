import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from numpy import linspace, meshgrid, c_
from matplotlib.cm import get_cmap
import time
from tqdm import tqdm


def plot_boundary(clf, X, y, grid_step=.01, poly_featurizer=None):
    x_min, x_max = X[:, 0].min() - .1, X[:, 0].max() + .1
    y_min, y_max = X[:, 1].min() - .1, X[:, 1].max() + .1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, grid_step),
                         np.arange(y_min, y_max, grid_step))

    # каждой точке в сетке [x_min, m_max]x[y_min, y_max]
    # ставим в соответствие свой цвет
    if poly_featurizer is not None:

        Z = clf.predict(poly_featurizer.transform(np.c_[xx.ravel(), yy.ravel()]))
    else:
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    for el in range(len(Z)):
        if Z[el] < 0.5:
            Z[el] = 0
        else:
            Z[el] = 1
    Z = Z.reshape(xx.shape)
    plt.contour(xx, yy, Z, cmap=plt.cm.Paired)
    plt.show()
    plt.savefig("output.png")

def plot_decision_boundary(X, y, model, steps=1000,poly_featurizer=None, cmap='Paired'):
    """
    Function to plot the decision boundary and data points of a model.
    Data points are colored based on their actual label.
    """
    cmap = get_cmap(cmap)
    print('Plot')
    # Define region of interest by data limits
    xmin, xmax = X[:,0].min() - 1, X[:,0].max() + 1
    ymin, ymax = X[:,1].min() - 1, X[:,1].max() + 1
    steps = 1000
    x_span = linspace(xmin, xmax, steps)
    y_span = linspace(ymin, ymax, steps)
    xx, yy = meshgrid(x_span, y_span)
    # Make predictions across region of interest
    labels=None
    if poly_featurizer is not None:
        labels = model.predict(poly_featurizer.transform(c_[xx.ravel(), yy.ravel()]))
    else:
        labels = model.predict(c_[xx.ravel(), yy.ravel()])
    for el in range(len(labels)):
        if labels[el] <0.5:
            labels[el]=0
        else: labels[el]=1

    # for i in tqdm(labels,
    #               desc="Plotting…",
    #               ascii=False, ncols=75):
    #     time.sleep(0.01)
    # Plot decision boundary in region of interest
    z = labels.reshape(xx.shape)

    fig, ax = plt.subplots()
    ax.contourf(xx, yy, z, cmap=cmap, alpha=0.5)

    # Get predicted labels on training data and plot
    #train_labels = model.predict(X)
    ax.scatter(X[:,0], X[:,1], c=y.ravel(), cmap=cmap, lw=0)

    return fig, ax


def convert_image(file):
  image = np.array(Image.open(file).convert('L'))
  return np.abs(((image / 255) - 1)*(-1))
