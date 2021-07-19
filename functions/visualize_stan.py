# visualize standardization approaches and data exploration
import matplotlib.pyplot as plt
import numpy as np
from functions.Load_data import load_the_pickle
from functions.Standardize import standardize


def sample_digits(data_location='data/train_points.p'):
    """
    :param data_location: data location
    :return: shows random sample of 25 images
    """
    train_labels, train_values = load_the_pickle(data_location)
    a = np.random.randint(0, 59975)
    b = a + 25
    fig = plt.figure()
    for i in range(a, b):
        images = train_values[i, :].reshape(28, 28)
        fig.add_subplot(5, 5, i - a + 1)
        plt.imshow(images, cmap='gray')
        plt.axis('off')

    plt.show()
    return


def compare_stan(data_location='data/train_points.p'):
    """

    :return:
    """
    fig = plt.figure(figsize=(10, 10))
    train_labels, train_values = load_the_pickle(data_location)
    for i in range(3):
        images = train_values[1:2, :]
        if i == 1:
            images = standardize(images, z=False)
        if i == 2:
            images = standardize(images)
        image = images.reshape(28, 28)
        fig.add_subplot(5, 5, i+1)
        plt.imshow(image, cmap='gray')
        plt.axis('off')
        plt.colorbar()

    plt.show()

    return
