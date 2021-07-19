import numpy as np
import pickle


def i_turned_myself_into_a_pickle_morty(data, new_data_location):
    """

    :param data: raw_data input
    :param new_data_location: where to save serialized data
    :return: saves binary form of data at new_data_location
    """
    x = np.genfromtxt(data, delimiter=',').astype(np.dtype('uint8'))
    with open(new_data_location, 'wb') as f:
        pickle.dump(x, f)


def load_the_pickle(pickled_data):
    """

    :param pickled_data: serialized data
    :return: unpacks binary data, labels and values are separately loaded
    """
    with open(pickled_data, 'rb') as f:
        x = pickle.load(f)
    labels = x[:, 0]
    values = x[:, 1:]
    return labels, values
