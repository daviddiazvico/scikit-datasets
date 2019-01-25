"""
LIBSVM datasets (https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets).

@author: David Diaz Vico
@license: MIT
"""

import numpy as np
import os
import scipy as sp
from sklearn.datasets import load_svmlight_file, load_svmlight_files
from sklearn.datasets.base import Bunch, get_data_home
from sklearn.model_selection import PredefinedSplit
from urllib.request import urlretrieve


BASE_URL = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets'
COLLECTIONS = {'binary', 'multiclass', 'regression', 'string'}


def _fetch_partition(collection, name, partition, dirname=None):
    """Fetch dataset partition."""
    filename = name.replace('/', '-') + partition
    url = BASE_URL + '/' + collection + '/' + name + partition
    try:
        f = filename + '.bz2' if dirname is None else os.path.join(dirname,
                                                                   filename + '.bz2')
        f, _ = urlretrieve(url + '.bz2', filename=f)
    except:
        try:
            f = filename if dirname is None else os.path.join(dirname, filename)
            f, _ = urlretrieve(url, filename=f)
        except:
            f = None
    return f


def _load(collection, name, dirname=None):
    """Load dataset."""
    filename = _fetch_partition(collection, name, '', dirname=dirname)
    filename_tr = _fetch_partition(collection, name, '.tr', dirname=dirname)
    filename_val = _fetch_partition(collection, name, '.val', dirname=dirname)
    filename_t = _fetch_partition(collection, name, '.t', dirname=dirname)
    filename_r = _fetch_partition(collection, name, '.r', dirname=dirname)
    if (filename_tr is not None) and (filename_val is not None) and (filename_t is not None):
        _, _, X_tr, y_tr, X_val, y_val, X_test, y_test = load_svmlight_files([filename,
                                                                              filename_tr,
                                                                              filename_val,
                                                                              filename_t])
        cv = PredefinedSplit([-1] * X_tr.shape[0] + [0] * X_val.shape[0])
        X = sp.sparse.vstack((X_tr, X_val))
        y = np.hstack((y_tr, y_val))
        X_remaining = y_remaining = None
    elif (filename_tr is not None) and (filename_val is not None):
        _, _, X_tr, y_tr, X_val, y_val = load_svmlight_files([filename,
                                                              filename_tr,
                                                              filename_val])
        cv = PredefinedSplit([-1] * X_tr.shape[0] + [0] * X_val.shape[0])
        X = sp.sparse.vstack((X_tr, X_val))
        y = np.hstack((y_tr, y_val))
        X_test = y_test = X_remaining = y_remaining = None
    elif (filename_t is not None) and (filename_r is not None):
        X, y, X_test, y_test, X_remaining, y_remaining = load_svmlight_files([filename,
                                                                              filename_t,
                                                                              filename_r])
        cv = None
    elif filename_t is not None:
        X, y, X_test, y_test = load_svmlight_files([filename, filename_t])
        X_remaining = y_remaining = cv = None
    else:
        X, y = load_svmlight_file(filename)
        X_test = y_test = X_remaining = y_remaining = cv = None
    return X, y, X_test, y_test, cv, X_remaining, y_remaining


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
    dirname = os.path.join(get_data_home(data_home=data_home), 'libsvm',
                           collection, name.replace('/', '-'))
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    X, y, X_test, y_test, cv, X_remaining, y_remaining = _load(collection, name,
                                                               dirname=dirname)
    data = Bunch(data=X, target=y, data_test=X_test, target_test=y_test,
                 inner_cv=cv, outer_cv=None, data_remaining=X_remaining,
                 target_remaining=y_remaining, DESCR=name)
    return data
