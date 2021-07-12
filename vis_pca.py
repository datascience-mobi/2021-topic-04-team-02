# visualize pca data
from final import *
import plotly.express as px
# loading data:
train_labels, train_values = load_the_pickle('data/train_points.p')
test_labels, test_values = load_the_pickle('data/test_points.p')

# standardization and PCA:
train_values_centered, train_mean = center(train_values)
train_values_pca, train_evs = PCA_func(train_values_centered, train_mean, number_of_pcs)

test_values_centered, test_mean = center(test_values, train_values)
test_values_pca, _ = PCA_func(test_values_centered,test_mean, number_of_pcs, train_evs=train_evs)


components_test3 = px.scatter_3d(test_values_pca, x=0, y=1, z=2, color=test_labels)
components_test3.update_traces(marker=dict(size=3))
components_train = px.scatter(train_values_pca, x=0, y=1, color=train_labels)
components_test = px.scatter(test_values_pca, x=0, y=1, color=test_labels)

