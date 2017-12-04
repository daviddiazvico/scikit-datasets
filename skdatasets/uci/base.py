"""
UCI datasets (https://archive.ics.uci.edu/ml/datasets.html).

@author: David Diaz Vico
@license: MIT
"""

import numpy as np
import pandas as pd

from ..base import Bunch, fetch_file


def load_train(name, url_data, url_names, names, target_names,
               return_X_y=False):
    """Load dataset.

    Load a dataset.

    Parameters
    ----------
    name: string
          Dataset name.
    url_data: string
              Dataset url.
    url_names: string
               Names url.
    names: list of strings
           Variable names.
    target_names: list of strings
                  Target names.
    return_X_y: bool, default=False
                If True, returns (data, target) instead of a Bunch object..

    Returns
    -------
    data: Bunch
          Dictionary-like object with all the data and metadata.
    X, y: arrays
          If return_X_y is True

    """
    feature_names = [n for n in names if not n in target_names]
    filename = fetch_file(name, url_data)
    data = pd.read_csv(filename, names=names)
    features = data[feature_names]
    categorical = features.select_dtypes(['object']).columns
    numerical = features.select_dtypes(['int64', 'float64']).columns
    try:
        features = pd.concat([pd.get_dummies(features[categorical]),
                              features[numerical]], axis=1).values
    except:
        features = features[numerical].values
    target = data[target_names].values

    if return_X_y:
        return features, target

    filename_descr = fetch_file(name, url_names)
    with open(filename_descr) as rst_file:
        fdescr = rst_file.read()

    return Bunch(data=features, target=target, target_names=target_names,
                 DESCR=fdescr, feature_names=feature_names)


def load_train_test(name, url_data, url_test, url_names, names, target_names,
                    return_X_y=False):
    """Load dataset with test partition.

    Load a dataset with test partition.

    Parameters
    ----------
    name: string
          Dataset name.
    url_data: string
              Dataset url.
    url_test: string
              Test dataset url.
    url_names: string
               Names url.
    names: list of strings
           Variable names.
    target_names: list of strings
                  Target names.
    return_X_y: bool, default=False
                If True, returns (data, target) instead of a Bunch object..

    Returns
    -------
    data: Bunch
          Dictionary-like object with all the data and metadata.
    (X, y), (X_test, y_test): lists of arrays
                              If return_X_y is True

    """
    feature_names = [n for n in names if not n in target_names]
    filename = fetch_file(name, url_data)
    data = pd.read_csv(filename, names=names, sep=', ', engine='python',
                       na_values='?')
    features = data[feature_names]
    categorical = features.select_dtypes(['object']).columns
    numerical = features.select_dtypes(['int64', 'float64']).columns
    features = pd.concat([pd.get_dummies(features[categorical]),
                          features[numerical]], axis=1).values
    target = pd.get_dummies(data[target_names],
                            drop_first=True).astype(np.int).values
    filename_test = fetch_file(name, url_test)
    data_test = pd.read_csv(filename_test, names=names, sep=', ',
                            engine='python', skiprows=1, na_values='?')
    features_test = data_test[feature_names]
    features_test = pd.concat([pd.get_dummies(features_test[categorical]),
                               features_test[numerical]], axis=1).values
    target_test = pd.get_dummies(data_test[target_names],
                                 drop_first=True).astype(np.int).values

    if return_X_y:
        return (features, target), (features_test, target_test)

    filename_descr = fetch_file(name, url_names)
    with open(filename_descr) as rst_file:
        fdescr = rst_file.read()

    return Bunch(data=features, target=target, data_test=features_test,
                 target_test=target_test, target_names=target_names,
                 DESCR=fdescr, feature_names=feature_names)
