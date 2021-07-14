import seaborn
import pandas as pd
import numpy
from functions.Load_data import load_the_pickle
from functions.PCA import PCA_func
import functions.KNN_predict as knn
from functions.Standardize import center
import itertools as itertools
import multiprocessing
from scipy.spatial import KDTree
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def knntest(traindataloaction,testdatalocation,kmin,kmax,pca_min,pca_max,new_data_location):
    """

    :param traindataloaction:
    :param testdatalocation:
    :param kmin:
    :param kmax:
    :param pca_min:
    :param pca_max:
    :param new_data_location:
    :return:
    """
    tests = []
    for number_of_pcs in range(pca_min, pca_max):

        # loading data:
        train_labels, train_values = load_the_pickle(traindataloaction)
        test_labels, test_values = load_the_pickle(testdatalocation)

        # standardization and PCA:
        train_values_centered, train_mean = center(train_values)
        train_values_pca, train_evs = PCA_func(train_values_centered, train_mean, number_of_pcs)

        test_values_centered, test_mean = center(test_values, Y=train_values)
        test_values_pca, _ = PCA_func(test_values_centered, test_mean, number_of_pcs, train_evs=train_evs)
        # kNN:
        tree = KDTree(train_values_pca)
        for k in range(kmin, kmax):
            hit = 0
            miss = 0
            if __name__ == '__main__':
                with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
                    result = p.starmap(knn.kdtree_knn, zip(test_values_pca[range(10000), :], itertools.repeat(k),
                                                           itertools.repeat(train_labels), itertools.repeat(tree)),
                                       chunksize=500)
                    for sample in range(10000):
                        if result[sample] == test_labels[sample]:
                            hit += 1
                        else:
                            miss += 1
                    print(hit, "vs", miss)
                    tests.append([number_of_pcs, k, hit])

    data = numpy.array(test)
    numpy.save(new_data_location, test2)

    return data

def knn_heatmap(datalocation,dropped_rows,dropped_columns):
    """

    :param datalocation:
    :return:
    """
    mydata = numpy.load(datalocation)
    X = numpy.array(mydata[:, 1])
    Y = numpy.array(mydata[:, 0])
    Z = numpy.array(mydata[:, 2]) / 10000
    df = pd.DataFrame.from_dict(numpy.array([X, Y, Z]).T)
    df.columns = ["k observed neighbours", "number of principle components", "Accuracy"]
    pivotted = df.pivot("k observed neighbours", "number of principle components", "Accuracy")
    pivotted = pd.DataFrame.drop(pivotted, index=dropped_rows, columns=dropped_columns)

    return seaborn.heatmap(pivotted)

def knn_3dplot(datalocation):
    """

    :param datalocation:
    :return:
    """

    mydata = numpy.load(datalocation)
    X = mydata[:, 1]
    Y = mydata[:, 0]
    Z = mydata[:, 2]

    fig = plt.figure()
    ax = Axes3D(fig,auto_add_to_figure=False)
    fig.add_axes(ax)
    ax.plot_trisurf(X, Y, Z,cmap="jet", linewidth=0.1)
    ax.set_xlabel("k observed neighbours")
    ax.set_ylabel("number of principle components")
    ax.set_zlabel("number of hits")
    ax.set_title("Accuracy plot")
    fig = plt.show()

    return fig