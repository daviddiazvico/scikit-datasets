"""
UCI datasets (https://archive.ics.uci.edu/ml/datasets.html).

@author: David Diaz Vico
@license: MIT
"""

import os

import numpy as np
from sklearn.utils import Bunch

from .base import fetch_file

BASE_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases'


def _load_csv(fname, **kwargs):
    """Load a csv file with targets in the last column and features in the rest.
    """
    data = np.genfromtxt(fname, dtype=str, delimiter=',', encoding=None,
                         **kwargs)
    X = data[:, :-1]
    y = data[:, -1]
    return X, y


def _fetch(name, data_home=None):
    """Fetch dataset."""
    subfolder = 'uci'
    filename = name + '.data'
    url = BASE_URL + '/' + name + '/' + filename

    filename = fetch_file(
        dataname=name,
        urlname=url,
        subfolder=subfolder,
        data_home=data_home,
    )
    X, y = _load_csv(filename)
    try:
        filename = name + '.test'
        url = BASE_URL + '/' + name + '/' + filename
        filename = fetch_file(
            dataname=name,
            urlname=url,
            subfolder=subfolder,
            data_home=data_home,
        )
        X_test, y_test = _load_csv(filename)
    except:
        X_test = y_test = None
    try:
        filename = name + '.names'
        url = BASE_URL + '/' + name + '/' + filename
        filename = fetch_file(
            dataname=name,
            urlname=url,
            subfolder=subfolder,
            data_home=data_home,
        )
    except:
        filename = name + '.info'
        url = BASE_URL + '/' + name + '/' + filename
        filename = fetch_file(
            dataname=name,
            urlname=url,
            subfolder=subfolder,
            data_home=data_home,
        )
    with open(filename) as rst_file:
        fdescr = rst_file.read()
    return X, y, X_test, y_test, fdescr


def fetch(name, data_home=None, *, return_X_y=False):
    """Fetch UCI dataset.

    Fetch a UCI dataset by name. More info at
    https://archive.ics.uci.edu/ml/datasets.html.

    Parameters
    ----------
    name : string
        Dataset name.
    data_home : string or None, default None
        Specify another download and cache folder for the data sets. By default
        all scikit-learn data is stored in ‘~/scikit_learn_data’ subfolders.
    return_X_y : bool, default=False
        If True, returns ``(data, target)`` instead of a Bunch object.

    Returns
    -------
    data : Bunch
        Dictionary-like object with all the data and metadata.

    (data, target) : tuple if ``return_X_y`` is True

    """
    X_train, y_train, X_test, y_test, DESCR = _fetch(name, data_home=data_home)

    if X_test is None or y_test is None:
        X = X_train
        y = y_train

        train_indices = None
        test_indices = None
    else:
        X = np.concatenate((X_train, X_test))
        y = np.concatenate((y_train, y_test))

        train_indices = list(range(len(X_train)))
        test_indices = list(range(len(X_train), len(X)))

    if return_X_y:
        return X, y

    return Bunch(
        data=X,
        target=y,
        train_indices=train_indices,
        validation_indices=[],
        test_indices=test_indices,
        inner_cv=None,
        outer_cv=None,
        DESCR=DESCR,
    )
