# visualize pca data
from final import *
import plotly.express as px
# loading data:
train_labels, train_values = load_the_pickle('data/train_points.p')
test_labels, test_values = load_the_pickle('data/test_points.p')

# standardization and PCA:
train_values_centered, train_mean = center(train_values)
train_values_pca, train_evs = PCA_func(train_values_centered, train_mean, number_of_pcs)

test_values_centered, test_mean = center_test_values(test_values, train_values)
test_values_pca, _ = PCA_func(test_values_centered,test_mean, number_of_pcs, train_evs=train_evs)


fig = px.scatter_3d(test_values_pca, x=0, y=1, z=2, color=test_labels)
fig.update_traces(marker=dict(size=3))
fig1 = px.scatter(train_values_pca, x=0, y=1, color=train_labels)
fig2 = px.scatter(test_values_pca, x=0, y=1, color=test_labels)
fig.show()
fig1.show()
fig2.show()