# visualize pca data
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt


def components_3d(test_values_pca, test_labels):
    """
    generates 3D plot using the values after pca colored depending on their label
    :param test_values_pca: input values
    :param test_labels: input labels determining the color
    :return: returns 3D plot
    """
    plot3d = px.scatter_3d(test_values_pca, x=0, y=1, z=2, color=test_labels)
    plot3d.update_traces(marker=dict(size=3))

    return plot3d.show()


def components_2d(values_pca, labels):
    """
    generates 2D plot using the values after pca colored depending on their label
    :param values_pca: input values
    :param labels: input labels determining the color
    :return: returns 2D plot
    """
    plot2d = px.scatter(values_pca, x=0, y=1, color=labels)

    return plot2d.show()


def inverse_pca(data, train_evs):
    data_reduced = np.dot(data - np.mean(train_evs), train_evs.T)
    data_reduced = data_reduced.reshape((28, 28))
    reduced_plot = plt.plot(data_reduced)

    return reduced_plot.show()
