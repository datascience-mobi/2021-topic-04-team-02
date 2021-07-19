# visualize pca data
import plotly.express as px


def components_3d(test_values_pca, test_labels):
    plot3d = px.scatter_3d(test_values_pca, x=0, y=1, z=2, color=test_labels)
    plot3d.update_traces(marker=dict(size=3))

    return plot3d.show()


def components_2d(values_pca, labels):
    plot2d = px.scatter(values_pca, x=0, y=1, color=labels)

    return plot2d.show()


