"""
Keras datasets (https://keras.io/datasets).

@author: David Diaz Vico
@license: MIT
"""

import numpy as np
from keras.datasets import (boston_housing, cifar10, cifar100, fashion_mnist,
                            imdb, mnist, reuters)
from sklearn.datasets.base import Bunch


DATASETS = {'boston_housing': boston_housing.load_data,
            'cifar10': cifar10.load_data, 'cifar100': cifar100.load_data,
            'fashion_mnist': fashion_mnist.load_data, 'imdb': imdb.load_data,
            'mnist': mnist.load_data, 'reuters': reuters.load_data}


def fetch_keras(name, **kwargs):
    """Fetch Keras dataset.

    Fetch a Keras dataset by name. More info at https://keras.io/datasets.

    Parameters
    ----------
    name : string
        Dataset name.
    **kwargs : dict
        Optional key-value arguments. See https://keras.io/datasets.

    Returns
    -------
    data : Bunch
        Dictionary-like object with all the data and metadata.

    """
    (X, y), (X_test, y_test) = DATASETS[name](**kwargs)
    if len(X.shape) > 2:
        name = name + ' ' + str(X.shape[1:]) + ' shaped'
        X_max = np.iinfo(X[0][0].dtype).max
        n_features = np.prod(X.shape[1:])
        X = X.reshape([X.shape[0], n_features]) / X_max
        X_test = X_test.reshape([X_test.shape[0], n_features]) / X_max
    return Bunch(data=X, target=y, data_test=X_test, target_test=y_test,
                 DESCR=name)
