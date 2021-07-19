# visualize pca data
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt


#def components_3d(test_values_pca, test_labels):
#    """
#    generates 3D plot using the values after pca colored depending on their label
#    :param test_values_pca: input values
#    :param test_labels: input labels determining the color
#    :return: returns 3D plot
#    """
#    plot3d = px.scatter_3d(test_values_pca, x=dimension 1, y=dimension 2, z=dimension 3, color=test_labels)
#    plot3d.update_traces(marker=dict(size=3))
#    plot3d.update_xaxes(title_text="Dimension 1")

#    return plot3d.show()


def components_3d(test_values_pca,labels,figure_title):
    mydata = test_values_pca
    x = mydata[:, 0]
    y = mydata[:, 1]
    z = mydata[:, 3]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    fig.add_axes(ax)
    plt.scatter(x, y, z,c=labels)
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_zlabel("PC 3")
    ax.set_title(figure_title)
    fig = plt.show()

    return fig

def components_2d(test_values_pca,labels, title, x_label, y_label):
    mydata = test_values_pca
    x = mydata[:, 0]
    y = mydata[:, 1]

    plt.scatter(x, y, c=labels)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    fig = plt.show()

    return fig


def components_reduced(data, train_evs):
    data_reduced = np.dot(data - np.mean(train_evs), train_evs.T)
    data_reduced = data_reduced.reshape((28, 28))
    reduced_plot = plt.plot(data_reduced)

    return reduced_plot.show()
