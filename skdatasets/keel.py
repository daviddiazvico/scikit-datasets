"""
Keel datasets (http://sci2s.ugr.es/keel).

@author: David Diaz Vico
@license: MIT
"""

import io
import numpy as np
import os
import pandas as pd
from sklearn.datasets.base import Bunch, get_data_home
from sklearn.model_selection import check_cv
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from urllib.request import urlretrieve
from zipfile import ZipFile


BASE_URL = 'http://sci2s.ugr.es/keel'
COLLECTIONS = {'classification', 'missing', 'imbalanced', 'multiInstance',
               'multilabel', 'textClassification', 'classNoise',
               'attributeNoise', 'semisupervised', 'regression', 'timeseries',
               'unsupervised', 'lowQuality'}


# WTFs
IMBALANCED_URLS = ['keel-dataset/datasets/imbalanced/imb_IRhigherThan9',
                   'keel-dataset/datasets/imbalanced/imb_IRhigherThan9p1',
                   'keel-dataset/datasets/imbalanced/imb_IRhigherThan9p2',
                   'keel-dataset/datasets/imbalanced/imb_IRhigherThan9p3',
                   'dataset/data/imbalanced',
                   'keel-dataset/datasets/imbalanced/imb_noisyBordExamples',
                   'keel-dataset/datasets/imbalanced/preprocessed']
IRREGULAR_DESCR_IMBALANCED_URLS = ['keel-dataset/datasets/imbalanced/imb_IRhigherThan9',
                                   'keel-dataset/datasets/imbalanced/imb_IRhigherThan9p1',
                                   'keel-dataset/datasets/imbalanced/imb_IRhigherThan9p2',
                                   'keel-dataset/datasets/imbalanced/imb_IRhigherThan9p3']
INCORRECT_DESCR_IMBALANCED_URLS = {'semisupervised': 'classification'}


def _load_Xy(zipfile, csvfile, sep=',', header=None, engine='python',
             na_values={'?'}, **kwargs):
    """Load a zippend csv file with targets in the last column and features in
       the rest."""
    with ZipFile(zipfile) as z:
        with z.open(csvfile) as c:
            s = io.StringIO(c.read().decode(encoding="utf8"))
            data = pd.read_csv(s, sep=sep, header=header, engine=engine,
                               na_values=na_values, **kwargs)
            X = pd.get_dummies(data.iloc[:, :-1])
            y = pd.factorize(data.iloc[:, -1].tolist(), sort=True)[0]
            return X, y


def _load_descr(collection, name, dirname=None):
    """Load a dataset description."""
    filename = name + '-names.txt'
    if collection == 'imbalanced':
        for url in IMBALANCED_URLS:
            if url in IRREGULAR_DESCR_IMBALANCED_URLS:
                url = BASE_URL + '/' + url + '/' + 'names' + '/' + filename
            else:
                url = BASE_URL + '/' + url + '/' + filename
            try:
                f = filename if dirname is None else os.path.join(dirname,
                                                                  filename)
                urlretrieve(url, filename=f)
                break
            except:
                continue
    else:
        collection = INCORRECT_DESCR_IMBALANCED_URLS[collection] if collection in INCORRECT_DESCR_IMBALANCED_URLS else collection
        url = BASE_URL + '/' + 'dataset/data' + '/' + collection + '/' + filename
        f = filename if dirname is None else os.path.join(dirname, filename)
        urlretrieve(url, filename=f)
    with open(f) as rst_file:
        fdescr = rst_file.read()
        nattrs = fdescr.count("@attribute")
    return nattrs, fdescr


def _fetch_keel_zip(collection, filename, dirname=None):
    """Fetch Keel dataset zip file."""
    if collection == 'imbalanced':
        for url in IMBALANCED_URLS:
            url = BASE_URL + '/' + url + '/' + filename
            try:
                f = filename if dirname is None else os.path.join(dirname,
                                                                  filename)
                urlretrieve(url, filename=f)
                break
            except:
                continue
    else:
        url = BASE_URL + '/' + 'dataset/data' + '/' + collection + '/' + filename
        f = filename if dirname is None else os.path.join(dirname, filename)
        urlretrieve(url, filename=f)
    return f


def _load_folds(collection, name, nfolds, dobscv, nattrs, dirname=None):
    """Load a dataset folds."""
    filename = name + '.zip'
    f = _fetch_keel_zip(collection, filename, dirname=dirname)
    X, y = _load_Xy(f, name + '.dat', skiprows=nattrs + 4)
    cv = None
    if nfolds in (5, 10):
        fold = 'dobscv' if dobscv else 'fold'
        filename = name + '-' + str(nfolds) + '-' + fold + '.zip'
        f = _fetch_keel_zip(collection, filename, dirname=dirname)
        Xs = []
        ys = []
        for i in range(nfolds):
            if dobscv:
                _name = os.path.join(name, name + '-' + str(nfolds) + 'dobscv-' + str(i + 1))
            else:
                _name = name + '-' + str(nfolds) + '-' + str(i + 1)
            _X, _y = _load_Xy(f, _name + 'tra.dat', skiprows=nattrs + 4)
            _X_test, _y_test = _load_Xy(f, _name + 'tst.dat',
                                        skiprows=nattrs + 4)
            Xs.append((_X, _X_test))
            ys.append((_y, _y_test))
        cv = check_cv(cv=Xs, y=ys)
    return X, y, cv


def fetch_keel(collection, name, data_home=None, nfolds=None, dobscv=False):
    """Fetch Keel dataset.

    Fetch a Keel dataset by collection and name. More info at
    http://sci2s.ugr.es/keel.

    Parameters
    ----------
    collection : string
        Collection name.
    name : string
        Dataset name.
    data_home : string or None, default None
        Specify another download and cache folder for the data sets. By default
        all scikit-learn data is stored in ‘~/scikit_learn_data’ subfolders.
    nfolds : int, default=None
        Number of folds. Depending on the dataset, valid values are
        {None, 1, 5, 10}.
    dobscv : bool, default=False
        If folds are in {5, 10}, indicates that the cv folds are distribution
        optimally balanced stratified. Only available for some datasets.
    **kwargs : dict
        Optional key-value arguments

    Returns
    -------
    data : Bunch
        Dictionary-like object with all the data and metadata.

    """
    if collection not in COLLECTIONS:
        raise Exception('Avaliable collections are ' + str(list(COLLECTIONS)))
    dirname = os.path.join(get_data_home(data_home=data_home), 'keel',
                           collection, name)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    nattrs, DESCR = _load_descr(collection, name, dirname=dirname)
    X, y, cv = _load_folds(collection, name, nfolds, dobscv, nattrs,
                           dirname=dirname)
    data = Bunch(data=X, target=y, outer_cv=cv, DESCR=DESCR)
    data = Bunch(**{k: v for k, v in data.items() if v is not None})
    return data
