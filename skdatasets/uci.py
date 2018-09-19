"""
UCI datasets (https://archive.ics.uci.edu/ml/datasets.html).

@author: David Diaz Vico
@license: MIT
"""

from sklearn.datasets.base import Bunch

import numpy as np
import pandas as pd

from .base import fetch_file


def _load_file(filename, names, feature_names, target_name):
    """Load a data file."""
    data = pd.read_csv(filename, names=names, sep=',', engine='python',
                       na_values='?')
    X = pd.get_dummies(data[feature_names]).values
    y = pd.factorize(data[target_name].tolist(), sort=True)[0]
    if (len(y.shape) == 1) or (np.prod(y.shape[1:]) == 1):
        y = y.ravel()
    return X, y


def _fetch(name, url_data, url_names, names, target_name, url_test=None):
    """Fetch dataset."""
    feature_names = [n for n in names if n != target_name]
    filename = fetch_file(name, url_data)
    X, y = _load_file(filename, names, feature_names, target_name)
    if url_test is not None:
        filename_test = fetch_file(name, url_test)
        X_test, y_test = _load_file(filename_test, names, feature_names,
                                    target_name)
    else:
        X_test = y_test = None
    filename_descr = fetch_file(name, url_names)
    with open(filename_descr) as rst_file:
        fdescr = rst_file.read()
    return (X, y, X_test, y_test, target_name, fdescr, feature_names)


BASE_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/'

datasets = {
    'abalone': {
        'args': [BASE_URL + 'abalone/abalone.data',
                 BASE_URL + 'abalone/abalone.names',
                 ['Sex', 'Length', 'Diameter', 'Height',
                  'Whole weight', 'Shucked weight',
                  'Viscera weight', 'Shell weight', 'Rings'],
                 'Rings']
        },
    'nursery': {
        'args': [BASE_URL + 'nursery/nursery.data',
                 BASE_URL + 'nursery/nursery.names',
                 ['parents', 'has_nurs', 'form', 'children',
                  'housing', 'finance', 'social', 'health',
                  'target'], 'target']
        },
    'adult': {
        'args': [BASE_URL + 'adult/adult.data',
                 BASE_URL + 'adult/adult.names',
                 ['age', 'workclass', 'fnlwgt', 'education',
                  'education-num', 'marital-status',
                  'occupation', 'relationship', 'race', 'sex',
                  'capital-gain', 'capital-loss',
                  'hours-per-week', 'native-country', 'target'],
                 'target',
                 BASE_URL + 'adult/adult.test']
        }
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
    (X, y, X_test, y_test, target_name,
     DESCR, feature_names) = _fetch(name, *datasets[name]['args'])
    if return_X_y:
        return X, y, X_test, y_test, None, None
    return Bunch(data=X, target=y, data_test=X_test, target_test=y_test,
                 target_names=target_name, DESCR=DESCR,
                 feature_names=feature_names)
