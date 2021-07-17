import numpy as np
import cv2
import matplotlib.pyplot as plt
from functions.Load_data import load_the_pickle
from functions.PCA import PCA_func
import functions.KNN_predict as knn
from functions.Standardize import center
from functions.Standardize import standardize


def form_filled_in(location_of_form):
    """

    :param location_of_form: filled in form as .png Example: "digits/Phone number form example.png"
    :return: array of images
    """
    y = 260
    h = 70
    w = 80

    # loading form in grayscale
    form = cv2.imread(location_of_form, cv2.IMREAD_GRAYSCALE)

    # 12 digits in phone number
    images = np.zeros((12, 784))
    for i in range(0, 12):
        # cropping form into individual images
        x = 130 + 89 * i
        crop_img = form[y:y + h, x:x + w]

        # lowering resolution to 28*28 pixels and inverting
        img = cv2.resize(255 - crop_img, (28, 28))

        # flattening to array
        flatten = img.flatten()
        images[i] = flatten
    return images


def call_me_maybe(images, x=12):
    """

    :param images: array with each digit as a row
    :param x: length of phone number, default is 12
    :return: phone number
    """
    phone_number = np.zeros(x, dtype=int)
    # number pcs and k
    number_of_pcs = 51
    k = 4

    # loading data:
    train_labels, train_values = load_the_pickle('data/train_points.p')
    test_values = images

    # standardization:
    train_values = standardize(train_values)
    test_values = standardize(test_values)

    # standardization and PCA:
    train_values_centered, train_mean = center(train_values)
    train_values_pca, train_evs = PCA_func(train_values_centered, train_mean, number_of_pcs)

    test_values_centered, test_mean = center(test_values, Y=train_values)
    test_values_pca, _ = PCA_func(test_values_centered, test_mean, number_of_pcs, train_evs=train_evs)

    for i in range(0, x):
        predicted_value = knn.weighted_knn("euclidean", trainvalues_pca=train_values_pca, x=test_values_pca[i, :], trainlabels=train_labels, k=k)
        phone_number[i] = predicted_value
    return phone_number


def show_phone_numbers(location_of_form):
    """

    :param location_of_form: filled in form as .png
    :return: shows phone number
    """
    fig = plt.figure(figsize=(10, 10))
    y = 260
    h = 70
    w = 80
    form = cv2.imread(location_of_form, cv2.IMREAD_GRAYSCALE)
    images = np.zeros((12, 784))
    for i in range(0, 12):
        x = 130 + 89 * i
        crop_img = form[y:y + h, x:x + w]
        img = cv2.resize(255 - crop_img, (28, 28))
        fig.add_subplot(1, 12, i + 1)
        plt.imshow(img, cmap='gray')
        plt.axis('off')
    return
