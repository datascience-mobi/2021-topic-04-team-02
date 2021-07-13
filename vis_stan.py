# visualize standardization approaches and data exploration
import matplotlib.pyplot as plt
import numpy as np
from functions.Load_data import load_the_pickle
from functions.PCA import PCA_func
from functions.KNN_predict import knn
from functions.Standardize import center


def sample_digits():
    """

    :return: shows random sample of 25 images
    """
    train_labels, train_values = load_the_pickle('data/train_points.p')
    fig = plt.figure(figsize=(10, 10))
    a = np.random.randint(0, 59975)
    b = a + 25
    fig = plt.figure()
    for i in range(a, b):
        mat_data = train_values[i, :].reshape(28, 28)
        fig.add_subplot(5, 5, i - a + 1)
        plt.imshow(mat_data, cmap='gray')
        plt.axis('off')

    plt.show()
    return


def compare_stan(n, ztransform):
    """

    :param n: number of images to run algorithm over
    :param ztransform: bool. z transform or center data
    :return: which digits incorrectly predicted and how many
    """

    # select number of principle components and k:
    number_of_pcs = 45
    k = 6

    # loading data:
    train_labels, train_values = load_the_pickle('data/train_points.p')
    test_labels, test_values = load_the_pickle('data/test_points.p')

    # standardization and PCA:
    train_values_centered, train_mean = center(train_values, scale=ztransform)
    train_values_pca, train_evs = PCA_func(train_values_centered, train_mean, number_of_pcs)

    test_values_centered, test_mean = center(test_values, Y=train_values, scale=ztransform)
    test_values_pca, _ = PCA_func(test_values_centered, test_mean, number_of_pcs, train_evs=train_evs)

    # kNN:
    hit = 0
    miss = 0
    missed_number = np.array([])

    for sample in range(n):
        predicted_value = knn("euclidean", trainvalues_pca=train_values_pca, X=test_values_pca[sample, :],
                              trainlabels=train_labels, k=k)
        labeled_value = test_labels[sample]
        if predicted_value == labeled_value:
            hit += 1
        else:
            miss += 1
            missed_number = np.append(missed_number, labeled_value)

    return missed_number, hit, miss


false_assignments_centered, hit_centered, miss_centered = compare_stan(10000, ztransform=False)
false_assignments_z_transformed, hit_z, miss_z = compare_stan(10000, ztransform=True)
