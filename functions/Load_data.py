import numpy as np
import pickle


def i_turned_myself_into_a_pickle_morty(data, new_data_location):
    x = np.genfromtxt(data, delimiter=',').astype(np.dtype('uint8'))
    with open(new_data_location, 'wb') as f:
        pickle.dump(x, f)
    # converts data to binary so we can share it on github


def load_the_pickle(pickled_data):
    with open(pickled_data, 'rb') as f:
        x = pickle.load(f)
    labels = x[:, 0]
    values = x[:, 1:]
    return labels, values
