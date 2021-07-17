import numpy as np


def any_na(data, replace=False, remove=False):
    """

    :param data: array
    :param replace: If True missing values replaced with 0
    :param remove: If True rows with missing values removed
    :return: data array
    """
    a = np.isnan(data)
    b = np.sum(a)
    if b != 0:
        print("There are " + str(b) + " missing values.")
        data_mod = data
        if replace:
            data_mod[a] = 0
            print("Replaced values with 0.")
            return data_mod
        if remove:
            data_mod = data_mod[~a.any(axis=1)]
            print("Removed rows with missing values.")
            return data_mod
        print("Consider replacing or removing these values.")
        return data
    print("There are no missing values in this data.")
    return


def rm_duplicates(data):
    """

    :param data: array
    :return: array with duplicate rows removed
    """
    data_mod = [tuple(row) for row in data]
    data_rm_dup = np.unique(data_mod, axis=0)
    if np.shape(data) != np.shape(data_rm_dup):
        print("Duplicates were removed.")
        return data_rm_dup
    else:
        print("No Duplicates.")
    return


def rm_range(data, upper=256, lower=0):
    """

    :param data: array to be tested whether all values within a range
    :param upper: upper range limit, number excluded. Default range is pixel intensity values.
    :param lower: lower range limit
    :return: rows of array removed that are out of range(lower, upper)
    """
    a = np.shape(data)
    if np.min(data) < lower:
        data = data[~np.any(data < lower, axis=1)]
    if np.max(data) >= upper:
        data = data[~np.any(data >= upper, axis=1)]
    if np.shape(data) != a:
        print("Rows with values out of range were removed.")
        return data
    else:
        print("No values out of range.")
        return
    return
