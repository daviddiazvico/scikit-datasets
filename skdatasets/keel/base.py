"""
Keel datasets (http://sci2s.ugr.es/keel/).

@author: David Diaz Vico
@license: MIT
"""

from os.path import join
import pandas as pd

from ..base import Bunch, fetch_file, fetch_zip


def _load_fold(fold, name, nfolds, data_home_fold, names, feature_names,
                target_names):
    """Load a dataset fold."""
    data = pd.read_csv(join(data_home_fold,
                            name + '-' + str(nfolds) + '-' + str(fold) + 'tra.dat'),
                       names=names, sep=', ', engine='python',
                       skiprows=len(names) + 4, na_values='?')
    features = pd.get_dummies(data[feature_names]).as_matrix()
    target = pd.factorize(data[target_names[0]].tolist(), sort=True)[0]
    data_test = pd.read_csv(join(data_home_fold,
                                 name + '-' + str(nfolds) + '-' + str(fold) + 'tst.dat'),
                            names=names, sep=', ', engine='python',
                            skiprows=len(names) + 4, na_values='?')
    features_test = pd.get_dummies(data_test[feature_names]).as_matrix()
    target_test = pd.factorize(data_test[target_names[0]].tolist(), sort=True)[0]
    return features, target, features_test, target_test


def _load_folds(name, nfolds, url, names, feature_names, target_names):
    """Load a dataset folds."""
    test_fold = []
    folds = []
    data_home_fold = fetch_zip(name, url + '/' + name + '-' + str(nfolds) + '-fold.zip')
    counter = 0
    for i in range(1, 6):
        _, _, features_test, _ = _load_fold(i, name, 5, data_home_fold, names,
                                             feature_names, target_names)
        folds.append(features_test)
        test_fold.extend([i - 1]*len(features_test))
        counter += len(features_test)
    return folds, test_fold


def load_imbalanced(name, url, names, target_names, return_X_y=False):
    """Load imbalanced dataset.

    Load an imabalanced dataset.

    Parameters
    ----------
    name: string
          Dataset name.
    url: string
         Dataset url.
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
    ((X, y), ): list of arrays
                If return_X_y is True

    """
    feature_names = [n for n in names if not n in target_names]

    data_home = fetch_zip(name, url + '/' + name + '.zip')
    data = pd.read_csv(join(data_home, name + '.dat'), names=names, sep=',\s*',
                       engine='python', skiprows=len(names) + 4, na_values='?')
    features = pd.get_dummies(data[feature_names]).as_matrix()
    target = pd.factorize(data[target_names[0]].tolist(), sort=True)[0]

    if return_X_y:
        return features, target

    folds, test_5fold = _load_folds(name, 5, url, names, feature_names,
                                    target_names)
    filename_descr = fetch_file(name, url + '/names/' + name + '-names.txt')
    with open(filename_descr) as rst_file:
        fdescr = rst_file.read()

    return Bunch(data=features, target=target, fold5=test_5fold,
                 target_names=target_names, DESCR=fdescr,
                 feature_names=feature_names)


def load_standard_classification(name, url, names, target_names,
                                 return_X_y=False):
    """Load standard classification dataset.

    Load a standard classification dataset.

    Parameters
    ----------
    name: string
          Dataset name.
    url: string
         Dataset url.
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
    ((X, y), ): list of arrays
                If return_X_y is True

    """
    feature_names = [n for n in names if not n in target_names]

    data_home = fetch_zip(name, url + '/' + name + '.zip')
    data = pd.read_csv(join(data_home, name + '.dat'), names=names, sep=',\s*',
                       engine='python', skiprows=len(names) + 4, na_values='?')
    features = pd.get_dummies(data[feature_names]).as_matrix()
    target = pd.factorize(data[target_names[0]].tolist(), sort=True)[0]

    if return_X_y:
        return features, target

    folds, test_5fold = _load_folds(name, 5, url, names, feature_names,
                                    target_names)
    folds, test_10fold = _load_folds(name, 10, url, names, feature_names,
                                     target_names)
    filename_descr = fetch_file(name, url + '/' + name + '-names.txt')
    with open(filename_descr) as rst_file:
        fdescr = rst_file.read()

    return Bunch(data=features, target=target, fold5=test_5fold,
                 fold10=test_10fold, target_names=target_names, DESCR=fdescr,
                 feature_names=feature_names)
