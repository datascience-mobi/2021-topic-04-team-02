# visualize pca data
from final import *
import plotly.express as px

fig = px.scatter_3d(test_values_pca, x=0, y=1, z=2, color=test_labels)
fig.update_traces(marker=dict(size=3))
fig1 = px.scatter(train_values_pca, x=0, y=1, color=train_labels)
fig2 = px.scatter(test_values_pca, x=0, y=1, color=test_labels)
fig.show()
fig1.show()
fig2.show()