"""
UCI datasets (https://archive.ics.uci.edu/ml/datasets.html).

@author: David Diaz Vico
@license: MIT
"""

import numpy as np
import os
from sklearn.datasets.base import Bunch, get_data_home
from urllib.request import urlretrieve


BASE_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases'


def _load_csv(fname, **kwargs):
    """Load a csv file with targets in the last column and features in the rest.
    """
    data = np.genfromtxt(fname, dtype=str, delimiter=',', encoding=None,
                         **kwargs)
    X = data[:, :-1]
    y = data[:, -1]
    return X, y


def _fetch(name, dirname=None):
    """Fetch dataset."""
    filename = name + '.data'
    url = BASE_URL + '/' + name + '/' + filename
    filename = filename if dirname is None else os.path.join(dirname, filename)
    urlretrieve(url, filename=filename)
    X, y = _load_csv(filename)
    try:
        filename = name + '.test'
        url = BASE_URL + '/' + name + '/' + filename
        filename = filename if dirname is None else os.path.join(dirname,
                                                                 filename)
        urlretrieve(url, filename=filename)
        X_test, y_test = _load_csv(filename)
    except:
        X_test = y_test = None
    try:
        filename = name + '.names'
        url = BASE_URL + '/' + name + '/' + filename
        filename = filename if dirname is None else os.path.join(dirname,
                                                                 filename)
        urlretrieve(url, filename=filename)
    except:
        filename = name + '.info'
        url = BASE_URL + '/' + name + '/' + filename
        filename = filename if dirname is None else os.path.join(dirname,
                                                                 filename)
        urlretrieve(url, filename=filename)
    with open(filename) as rst_file:
        fdescr = rst_file.read()
    return X, y, X_test, y_test, fdescr


def fetch_uci(name, data_home=None):
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

    Returns
    -------
    data : Bunch
        Dictionary-like object with all the data and metadata.

    """
    dirname = os.path.join(get_data_home(data_home=data_home), 'uci', name)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    X, y, X_test, y_test, DESCR = _fetch(name, dirname=dirname)
    data = Bunch(data=X, target=y, data_test=X_test, target_test=y_test,
                 DESCR=DESCR)
    data = Bunch(**{k: v for k, v in data.items() if v is not None})
    return data
