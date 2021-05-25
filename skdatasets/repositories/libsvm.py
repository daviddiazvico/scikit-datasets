"""
LIBSVM datasets (https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets).

@author: David Diaz Vico
@license: MIT
"""

import os
from urllib.error import HTTPError

import numpy as np
from sklearn.datasets import load_svmlight_file, load_svmlight_files
from sklearn.model_selection import PredefinedSplit
from sklearn.utils import Bunch

import scipy as sp

from .base import fetch_file

BASE_URL = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets'
COLLECTIONS = {'binary', 'multiclass', 'regression', 'string'}


def _fetch_partition(collection, name, partition, data_home=None):
    """Fetch dataset partition."""
    subfolder = os.path.join('libsvm', collection)
    dataname = name.replace('/', '-')

    url = BASE_URL + '/' + collection + '/' + name + partition
    try:
        f = fetch_file(
            dataname,
            urlname=url + '.bz2',
            subfolder=subfolder,
            data_home=data_home,
        )
    except HTTPError:
        try:
            f = fetch_file(
                dataname,
                urlname=url,
                subfolder=subfolder,
                data_home=data_home,
            )
        except:
            f = None

    if f is not None:
        f = os.fspath(f)

    return f


def _load(collection, name, data_home=None):
    """Load dataset."""
    filename = _fetch_partition(collection, name, '', data_home)
    filename_tr = _fetch_partition(collection, name, '.tr', data_home)
    filename_val = _fetch_partition(collection, name, '.val', data_home)
    filename_t = _fetch_partition(collection, name, '.t', data_home)
    filename_r = _fetch_partition(collection, name, '.r', data_home)

    if (filename_tr is not None) and (filename_val is not None) and (filename_t is not None):

        _, _, X_tr, y_tr, X_val, y_val, X_test, y_test = load_svmlight_files([
            filename,
            filename_tr,
            filename_val,
            filename_t,
        ])

        cv = PredefinedSplit([-1] * X_tr.shape[0] + [0] * X_val.shape[0])

        X = sp.sparse.vstack((X_tr, X_val, X_test))
        y = np.hstack((y_tr, y_val, y_test))

        # Compute indices
        train_indices = list(range(X_tr.shape[0]))
        validation_indices = list(range(
            X_tr.shape[0],
            X_tr.shape[0] + X_val.shape[0],
        ))
        test_indices = list(range(X_tr.shape[0] + X_val.shape[0], X.shape[0]))

    elif (filename_tr is not None) and (filename_val is not None):

        _, _, X_tr, y_tr, X_val, y_val = load_svmlight_files([
            filename,
            filename_tr,
            filename_val,
        ])

        cv = PredefinedSplit([-1] * X_tr.shape[0] + [0] * X_val.shape[0])

        X = sp.sparse.vstack((X_tr, X_val))
        y = np.hstack((y_tr, y_val))

        # Compute indices
        train_indices = list(range(X_tr.shape[0]))
        validation_indices = list(range(X_tr.shape[0], X.shape[0]))
        test_indices = []

    elif (filename_t is not None) and (filename_r is not None):

        X_tr, y_tr, X_test, y_test, X_remaining, y_remaining = load_svmlight_files([
            filename,
            filename_t,
            filename_r,
        ])

        X = sp.sparse.vstack((X_tr, X_test, X_remaining))
        y = np.hstack((y_tr, y_test, y_remaining))

        # Compute indices
        train_indices = list(range(X_tr.shape[0]))
        validation_indices = []
        test_indices = list(range(
            X_tr.shape[0], X_tr.shape[0] + X_test.shape[0]
        ))

        cv = None

    elif filename_t is not None:

        X_tr, y_tr, X_test, y_test = load_svmlight_files([
            filename,
            filename_t,
        ])

        X = sp.sparse.vstack((X_tr, X_test))
        y = np.hstack((y_tr, y_test))

        # Compute indices
        train_indices = list(range(X_tr.shape[0]))
        validation_indices = []
        test_indices = list(range(X_tr.shape[0], X.shape[0]))

        cv = None

    else:

        X, y = load_svmlight_file(filename)

        # Compute indices
        train_indices = []
        validation_indices = []
        test_indices = []

        cv = None

    return X, y, train_indices, validation_indices, test_indices, cv


def fetch(collection, name, data_home=None):
    """Fetch LIBSVM dataset.

    Fetch a LIBSVM dataset by collection and name. More info at
    https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets.

    Parameters
    ----------
    collection : string
        Collection name.
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
    if collection not in COLLECTIONS:
        raise Exception('Avaliable collections are ' + str(list(COLLECTIONS)))

    X, y, train_indices, validation_indices, test_indices, cv = _load(
        collection,
        name,
        data_home=data_home,
    )

    data = Bunch(
        data=X,
        target=y,
        train_indices=train_indices,
        validation_indices=validation_indices,
        test_indices=test_indices,
        inner_cv=cv,
        outer_cv=None,
        DESCR=name,
    )

    return data
