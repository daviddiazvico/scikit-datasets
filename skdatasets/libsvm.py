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
from sklearn.model_selection import check_cv, PredefinedSplit
from urllib.request import urlretrieve


BASE_URL = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets'
COLLECTIONS = {'binary', 'multiclass', 'regression', 'string'}


def _fetch_partition(collection, name, partition, dirname=None):
    """Fetch dataset partition."""
    filename = name.replace('/', '-') + partition
    url = BASE_URL + '/' + collection + '/' + name
    try:
        f = filename + '.bz2' if dirname is None else os.path.join(dirname,
                                                                   filename + '.bz2')
        urlretrieve(url + '.bz2', filename=f)
    except:
        f = filename if dirname is None else os.path.join(dirname, filename)
        urlretrieve(url, filename=f)
    return f


def _load(collection, name, dirname=None):
    """Load dataset."""
    filename = _fetch_partition(collection, name, '', dirname=dirname)
    try:
        filename_tr = _fetch_partition(collection, name, '.tr', dirname=dirname)
    except:
        filename_tr = None
    try:
        filename_val = _fetch_partition(collection, name, '.val',
                                        dirname=dirname)
    except:
        filename_val = None
    try:
        filename_t = _fetch_partition(collection, name, '.t', dirname=dirname)
    except:
        filename_t = None
    try:
        filename_r = _fetch_partition(collection, name, '.r', dirname=dirname)
    except:
        filename_r = None
    if (filename_tr is not None) and (filename_val is not None) and (filename_t is not None):
        _, _, X_tr, y_tr, X_val, y_val, X_test, y_test = load_svmlight_files([filename,
                                                                              filename_tr,
                                                                              filename_val,
                                                                              filename_t])
        X = sp.sparse.vstack((X_tr, X_val))
        y = np.hstack((y_tr, y_val))
        inner_cv = PredefinedSplit([item for sublist in [[-1] * X_tr.shape[0], [0] * X_val.shape[0]] for item in sublist])
        outer_cv = check_cv(cv=(X, X_test), y=(y, y_test))
        X_remaining = y_remaining = None
    elif (filename_tr is not None) and (filename_val is not None):
        _, _, X_tr, y_tr, X_val, y_val = load_svmlight_files([filename,
                                                              filename_tr,
                                                              filename_val])
        X = sp.sparse.vstack((X_tr, X_val))
        y = np.hstack((y_tr, y_val))
        inner_cv = PredefinedSplit([item for sublist in [[-1] * X_tr.shape[0], [0] * X_val.shape[0]] for item in sublist])
        outer_cv = X_remaining = y_remaining = None
    elif (filename_t is not None) and (filename_r is not None):
        X, y, X_test, y_test, X_remaining, y_remaining = load_svmlight_files([filename,
                                                                              filename_t,
                                                                              filename_r])
        inner_cv = None
        outer_cv = check_cv(cv=(X, X_test), y=(y, y_test))
    elif filename_t is not None:
        X, y, X_test, y_test = load_svmlight_files([filename, filename_t])
        inner_cv = X_remaining = y_remaining = None
        outer_cv = check_cv(cv=(X, X_test), y=(y, y_test))
    else:
        X, y = load_svmlight_file(filename)
        inner_cv = outer_cv = X_remaining = y_remaining = None
    return X, y, inner_cv, outer_cv, X_remaining, y_remaining


def fetch_libsvm(collection, name, data_home=None):
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
    X, y, inner_cv, outer_cv, X_remaining, y_remaining = _load(collection, name,
                                                               dirname=dirname)
    data = Bunch(data=X, target=y, inner_cv=inner_cv, outer_cv=outer_cv,
                 data_remaining=X_remaining, target_remaining=y_remaining,
                 DESCR=name)
    return data
