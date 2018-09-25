"""
Keras datasets
(https://keras.io/datasets/).

@author: David Diaz Vico
@license: MIT
"""

from sklearn.datasets.base import Bunch

from keras.datasets import (boston_housing, cifar10, cifar100, fashion_mnist,
                            imdb, mnist, reuters)
import numpy as np

datasets = {
    'boston_housing': {'loader': boston_housing.load_data, 'pixel_max': None},
    'cifar10': {'loader': cifar10.load_data, 'pixel_max': 256.0},
    'cifar100': {'loader': cifar100.load_data, 'pixel_max': 256.0},
    'fashion_mnist': {'loader': fashion_mnist.load_data, 'pixel_max': 256.0},
    'imdb': {'loader': imdb.load_data, 'pixel_max': None},
    'mnist': {'loader': mnist.load_data, 'pixel_max': 256.0},
    'reuters': {'loader': reuters.load_data, 'pixel_max': None}
    }


def load(name, return_X_y=False):
    """Load dataset.

    Load a dataset.

    Parameters
    ----------
    name: string
          Dataset name.
    return_X_y: bool, default=False
                If True, returns (data, target) instead of a Bunch object.

    Returns
    -------
    data: Bunch
          Dictionary-like object with all the data and metadata.
    X, y, X_test, y_test, inner_cv, outer_cv: arrays
                                              If return_X_y is True

    """
    (X, y), (X_test, y_test) = datasets[name]['loader']()
    if datasets[name]['pixel_max'] is not None:
        X = (X.reshape([X.shape[0], np.prod(X.shape[1:])]) /
             datasets[name]['pixel_max'])
        X_test = (X_test.reshape(
            [X_test.shape[0], np.prod(X_test.shape[1:])]) /
            datasets[name]['pixel_max'])
    if (len(y.shape) == 1) or (np.prod(y.shape[1:]) == 1):
        y = y.flatten()
    if (len(y_test.shape) == 1) or (np.prod(y_test.shape[1:]) == 1):
        y_test = y_test.flatten()
    if return_X_y:
        return X, y, X_test, y_test, None, None
    return Bunch(data=X, target=y, data_test=X_test, target_test=y_test,
                 DESCR=name)
