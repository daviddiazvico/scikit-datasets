"""
LIBSVM datasets (https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/).

@author: David Diaz Vico
@license: MIT
"""

from sklearn.datasets import load_svmlight_file, load_svmlight_files
from sklearn.datasets.base import Bunch
from sklearn.model_selection import PredefinedSplit

from .base import fetch_bz2, fetch_file


def _load_train(name, url, fetch_file=fetch_file):
    """Load dataset."""
    filename = fetch_file(name, url)
    X, y = load_svmlight_file(filename)
    X = X.todense()
    y[y == -1] = 0
    return X, y, None, None, None


def _load_train_scale(name, url, url_scale, fetch_file=fetch_file):
    """Load dataset with scaled version."""
    filename = fetch_file(name, url)
    filename_scale = fetch_file(name, url_scale)
    X, y, X_scale, y_scale = load_svmlight_files([filename, filename_scale])
    X = X.todense()
    y[y == -1] = 0
    return X, y, None, None, None


def _load_train_test(name, url, url_test, fetch_file=fetch_file):
    """Load dataset with test partition."""
    filename = fetch_file(name, url)
    filename_test = fetch_file(name, url_test)
    X, y, X_test, y_test = load_svmlight_files([filename, filename_test])
    X = X.todense()
    X_test = X_test.todense()
    y[y == -1] = 0
    y_test[y_test == -1] = 0
    return X, y, X_test, y_test, None


def _load_train_test_remaining(name, url, url_test, url_remaining,
                               fetch_file=fetch_file):
    """Load dataset with test and remaining partitions."""
    filename = fetch_file(name, url)
    filename_test = fetch_file(name, url_test)
    filename_remaining = fetch_file(name, url_remaining)
    X, y, X_test, y_test, X_remaining, y_remaining = load_svmlight_files([filename,
                                                                          filename_test,
                                                                          filename_remaining])
    X = X.todense()
    X_test = X_test.todense()
    y[y == -1] = 0
    y_test[y_test == -1] = 0
    return X, y, X_test, y_test, None


def _load_train_val_test(name, url, url_tr, url_val, url_test,
                         fetch_file=fetch_file):
    """Load dataset with train, validation and test partitions."""
    filename = fetch_file(name, url)
    filename_tr = fetch_file(name, url_tr)
    filename_val = fetch_file(name, url_val)
    filename_test = fetch_file(name, url_test)
    X, y, X_tr, y_tr, X_val, y_val, X_test, y_test = load_svmlight_files(
        [filename, filename_tr, filename_val, filename_test])
    X = X.todense()
    X_test = X_test.todense()
    y[y == -1] = 0
    y_test[y_test == -1] = 0
    inner_cv = PredefinedSplit(
        [item for sublist in [[-1] * X_tr.shape[0],
                              [0] * X_val.shape[0]] for item in sublist])
    return X, y, X_test, y_test, inner_cv


def _load_train_test_scale(name, url, url_test, url_scale, url_test_scale,
                           fetch_file=fetch_file):
    """Load dataset with test partition and scaled version."""
    filename = fetch_file(name, url)
    filename_test = fetch_file(name, url_test)
    filename_scale = fetch_file(name, url_scale)
    filename_scale_test = fetch_file(name, url_test_scale)
    (X, y, X_test, y_test, X_scale,
     y_scale, X_scale_test, y_scale_test) = load_svmlight_files(
         [filename, filename_test, filename_scale, filename_scale_test])
    X = X.todense()
    X_test = X_test.todense()
    y[y == -1] = 0
    y_test[y_test == -1] = 0
    return X, y, X_test, y_test, None


BASE_URL = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/'

datasets = {
    'skin_nonskin': {
        'loader': _load_train,
        'args': [BASE_URL + 'binary/skin_nonskin']
        },
    'australian': {
        'loader': _load_train_scale,
        'args': [BASE_URL + 'binary/australian',
                 BASE_URL + 'binary/australian_scale']
        },
    'covtype.binary': {
        'loader': _load_train_scale,
        'args': [BASE_URL + 'binary/covtype.libsvm.binary.bz2',
                 BASE_URL + 'binary/covtype.libsvm.binary.scale.bz2',
                 fetch_bz2]
        },
    'diabetes': {
        'loader': _load_train_scale,
        'args': [BASE_URL + 'binary/diabetes',
                 BASE_URL + 'binary/diabetes_scale']
        },
    'german.numer': {
        'loader': _load_train_scale,
        'args': [BASE_URL + 'binary/german.numer',
                 BASE_URL + 'binary/german.numer_scale']
        },
    'heart': {
        'loader': _load_train_scale,
        'args': [BASE_URL + 'binary/heart',
                 BASE_URL + 'binary/heart_scale']
        },
    'a4a': {
        'loader': _load_train_test,
        'args': [BASE_URL + 'binary/a4a',
                 BASE_URL + 'binary/a4a.t']
        },
    'a8a': {
        'loader': _load_train_test,
        'args': [BASE_URL + 'binary/a8a',
                 BASE_URL + 'binary/a8a.t']
        },
    'epsilon': {
        'loader': _load_train_test,
        'args': [BASE_URL + 'binary/epsilon_normalized.bz2',
                 BASE_URL + 'binary/epsilon_normalized.t.bz2',
                 fetch_bz2]
        },
    'pendigits': {
        'loader': _load_train_test,
        'args': [BASE_URL + 'multiclass/pendigits',
                 BASE_URL + 'multiclass/pendigits.t']
        },
    'usps': {
        'loader': _load_train_test,
        'args': [BASE_URL + 'multiclass/usps.bz2',
                 BASE_URL + 'multiclass/usps.t.bz2',
                 fetch_bz2]
        },
    'w7a': {
        'loader': _load_train_test,
        'args': [BASE_URL + 'binary/w7a',
                 BASE_URL + 'binary/w7a.t']
        },
    'w8a': {
        'loader': _load_train_test,
        'args': [BASE_URL + 'binary/w8a',
                 BASE_URL + 'binary/w8a.t']
        },
    'cod-rna': {
        'loader': _load_train_test_remaining,
        'args': [BASE_URL + 'binary/cod-rna',
                 BASE_URL + 'binary/cod-rna.t',
                 BASE_URL + 'binary/cod-rna.r']
        },
    'combined': {
        'loader': _load_train_test_scale,
        'args': [BASE_URL + 'multiclass/vehicle/combined.bz2',
                 BASE_URL + 'multiclass/vehicle/combined.t.bz2',
                 BASE_URL + 'multiclass/vehicle/combined_scale.bz2',
                 BASE_URL + 'multiclass/vehicle/combined_scale.t.bz2',
                 fetch_bz2]
        },
    'news20': {
        'loader': _load_train_test_scale,
        'args': [BASE_URL + 'multiclass/news20.bz2',
                 BASE_URL + 'multiclass/news20.t.bz2',
                 BASE_URL + 'multiclass/news20.scale.bz2',
                 BASE_URL + 'multiclass/news20.t.scale.bz2',
                 fetch_bz2]
        },
    'dna': {
        'loader': _load_train_val_test,
        'args': [BASE_URL + 'multiclass/dna.scale',
                 BASE_URL + 'multiclass/dna.scale.tr',
                 BASE_URL + 'multiclass/dna.scale.val',
                 BASE_URL + 'multiclass/dna.scale.t']
        },
    'ijcnn1': {
        'loader': _load_train_val_test,
        'args': [BASE_URL + 'binary/ijcnn1.bz2',
                 BASE_URL + 'binary/ijcnn1.tr.bz2',
                 BASE_URL + 'binary/ijcnn1.val.bz2',
                 BASE_URL + 'binary/ijcnn1.t.bz2',
                 fetch_bz2]
        },
    'letter': {
        'loader': _load_train_val_test,
        'args': [BASE_URL + 'multiclass/letter.scale',
                 BASE_URL + 'multiclass/letter.scale.tr',
                 BASE_URL + 'multiclass/letter.scale.val',
                 BASE_URL + 'multiclass/letter.scale.t']
        },
    'satimage': {
        'loader': _load_train_val_test,
        'args': [BASE_URL + 'multiclass/satimage.scale',
                 BASE_URL + 'multiclass/satimage.scale.tr',
                 BASE_URL + 'multiclass/satimage.scale.val',
                 BASE_URL + 'multiclass/satimage.scale.t']
        },
    'shuttle': {
        'loader': _load_train_val_test,
        'args': [BASE_URL + 'multiclass/shuttle.scale',
                 BASE_URL + 'multiclass/shuttle.scale.tr',
                 BASE_URL + 'multiclass/shuttle.scale.val',
                 BASE_URL + 'multiclass/shuttle.scale.t']
        },
    'abalone': {
        'loader': _load_train_scale,
        'args': [BASE_URL + 'regression/abalone',
                 BASE_URL + 'regression/abalone_scale']
        },
    'bodyfat': {
        'loader': _load_train_scale,
        'args': [BASE_URL + 'regression/bodyfat',
                 BASE_URL + 'regression/bodyfat_scale']
        },
    'cadata': {
        'loader': _load_train,
        'args': [BASE_URL + 'regression/cadata']
        },
    'cpusmall': {
        'loader': _load_train_scale,
        'args': [BASE_URL + 'regression/cpusmall',
                 BASE_URL + 'regression/cpusmall_scale']
        },
    'housing': {
        'loader': _load_train_scale,
        'args': [BASE_URL + 'regression/housing',
                 BASE_URL + 'regression/housing_scale']
        },
    'mg': {
        'loader': _load_train_scale,
        'args': [BASE_URL + 'regression/mg',
                 BASE_URL + 'regression/mg_scale']
        },
    'mpg': {
        'loader': _load_train_scale,
        'args': [BASE_URL + 'regression/mpg',
                 BASE_URL + 'regression/mpg_scale']
        },
    'pyrim': {
        'loader': _load_train_scale,
        'args': [BASE_URL + 'regression/pyrim',
                 BASE_URL + 'regression/pyrim_scale']
        },
    'space_ga': {
        'loader': _load_train_scale,
        'args': [BASE_URL + 'regression/space_ga',
                 BASE_URL + 'regression/space_ga_scale']
        },
    'triazines': {
        'loader': _load_train_scale,
        'args': [BASE_URL + 'regression/triazines',
                 BASE_URL + 'regression/triazines_scale']
        },
    'E2006-log1p': {
        'loader': _load_train_scale,
        'args': [BASE_URL + 'regression/log1p.E2006.train.bz2',
                 BASE_URL + 'regression/log1p.E2006.test.bz2',
                 fetch_bz2]
        },
    'E2006-tfidf': {
        'loader': _load_train_scale,
        'args': [BASE_URL + 'regression/E2006.train.bz2',
                 BASE_URL + 'regression/E2006.test.bz2',
                 fetch_bz2]
        },
    'YearPredictionMSD': {
        'loader': _load_train_scale,
        'args': [BASE_URL + 'regression/YearPredictionMSD.bz2',
                 BASE_URL + 'regression/YearPredictionMSD.t.bz2',
                 fetch_bz2]
        }
    }


def load(name, return_X_y=False):
    """Load

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
    X, y, X_test, y_test, inner_cv = datasets[name]['loader'](
        name, *datasets[name]['args'])
    if return_X_y:
        return X, y, X_test, y_test, inner_cv, None
    return Bunch(data=X, target=y, data_test=X_test, target_test=y_test,
                 inner_cv=inner_cv, DESCR=name)
