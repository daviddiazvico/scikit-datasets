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
    X, y, X_tr, y_tr, X_val, y_val, X_test, y_test = load_svmlight_files([filename,
                                                                          filename_tr,
                                                                          filename_val,
                                                                          filename_test])
    X = X.todense()
    X_test = X_test.todense()
    y[y == -1] = 0
    y_test[y_test == -1] = 0
    inner_cv = PredefinedSplit([item for sublist in [[-1] * X_tr.shape[0], [0] * X_val.shape[0]] for item in sublist])
    return X, y, X_test, y_test, inner_cv


def _load_train_test_scale(name, url, url_test, url_scale, url_test_scale,
                           fetch_file=fetch_file):
    """Load dataset with test partition and scaled version."""
    filename = fetch_file(name, url)
    filename_test = fetch_file(name, url_test)
    filename_scale = fetch_file(name, url_scale)
    filename_scale_test = fetch_file(name, url_test_scale)
    X, y, X_test, y_test, X_scale, y_scale, X_scale_test, y_scale_test = load_svmlight_files([filename,
                                                                                              filename_test,
                                                                                              filename_scale,
                                                                                              filename_scale_test])
    X = X.todense()
    X_test = X_test.todense()
    y[y == -1] = 0
    y_test[y_test == -1] = 0
    return X, y, X_test, y_test, None


datasets = {'skin_nonskin': {'loader': _load_train,
                             'args': ['https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/skin_nonskin']},
            'australian': {'loader': _load_train_scale,
                           'args': ['https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/australian',
                                    'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/australian_scale']},
            'covtype.binary': {'loader': _load_train_scale,
                               'args': ['https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/covtype.libsvm.binary.bz2',
                                        'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/covtype.libsvm.binary.scale.bz2',
                                        fetch_bz2]},
            'diabetes': {'loader': _load_train_scale,
                         'args': ['https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/diabetes',
                                  'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/diabetes_scale']},
            'german.numer': {'loader': _load_train_scale,
                             'args': ['https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/german.numer',
                                      'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/german.numer_scale']},
            'heart': {'loader': _load_train_scale,
                      'args': ['https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/heart',
                               'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/heart_scale']},
            'a4a': {'loader': _load_train_test,
                    'args': ['http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a4a',
                             'http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a4a.t']},
            'a8a': {'loader': _load_train_test,
                    'args': ['http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a8a',
                             'http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a8a.t']},
            'epsilon': {'loader': _load_train_test,
                        'args': ['https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/epsilon_normalized.bz2',
                                 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/epsilon_normalized.t.bz2',
                                 fetch_bz2]},
            'pendigits': {'loader': _load_train_test,
                          'args': ['https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/pendigits',
                                   'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/pendigits.t']},
            'usps': {'loader': _load_train_test,
                     'args': ['https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/usps.bz2',
                              'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/usps.t.bz2',
                              fetch_bz2]},
            'w7a': {'loader': _load_train_test,
                    'args': ['http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/w7a',
                             'http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/w7a.t']},
            'w8a': {'loader': _load_train_test,
                    'args': ['http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/w8a',
                             'http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/w8a.t']},
            'cod-rna': {'loader': _load_train_test_remaining,
                        'args': ['https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/cod-rna',
                                 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/cod-rna.t',
                                 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/cod-rna.r']},
            'combined': {'loader': _load_train_test_scale,
                         'args': ['https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/vehicle/combined.bz2',
                                  'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/vehicle/combined.t.bz2',
                                  'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/vehicle/combined_scale.bz2',
                                  'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/vehicle/combined_scale.t.bz2',
                                  fetch_bz2]},
            'news20': {'loader': _load_train_test_scale,
                       'args': ['https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/news20.bz2',
                                'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/news20.t.bz2',
                                'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/news20.scale.bz2',
                                'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/news20.t.scale.bz2',
                                fetch_bz2]},
            'dna': {'loader': _load_train_val_test,
                    'args': ['https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/dna.scale',
                             'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/dna.scale.tr',
                             'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/dna.scale.val',
                             'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/dna.scale.t']},
            'ijcnn1': {'loader': _load_train_val_test,
                       'args': ['https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/ijcnn1.bz2',
                                'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/ijcnn1.tr.bz2',
                                'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/ijcnn1.val.bz2',
                                'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/ijcnn1.t.bz2',
                                fetch_bz2]},
            'letter': {'loader': _load_train_val_test,
                       'args': ['https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/letter.scale',
                                'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/letter.scale.tr',
                                'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/letter.scale.val',
                                'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/letter.scale.t']},
            'satimage': {'loader': _load_train_val_test,
                         'args': ['https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/satimage.scale',
                                  'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/satimage.scale.tr',
                                  'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/satimage.scale.val',
                                  'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/satimage.scale.t']},
            'shuttle': {'loader': _load_train_val_test,
                        'args': ['https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/shuttle.scale',
                                 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/shuttle.scale.tr',
                                 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/shuttle.scale.val',
                                 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/shuttle.scale.t']},
            'abalone': {'loader': _load_train_scale,
                        'args': ['https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/abalone',
                                 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/abalone_scale']},
            'bodyfat': {'loader': _load_train_scale,
                        'args': ['https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/bodyfat',
                                 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/bodyfat_scale']},
            'cadata': {'loader': _load_train,
                       'args': ['https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/cadata']},
            'cpusmall': {'loader': _load_train_scale,
                         'args': ['https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/cpusmall',
                                  'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/cpusmall_scale']},
            'housing': {'loader': _load_train_scale,
                        'args': ['https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/housing',
                                 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/housing_scale']},
            'mg': {'loader': _load_train_scale,
                   'args': ['https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/mg',
                            'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/mg_scale']},
            'mpg': {'loader': _load_train_scale,
                    'args': ['https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/mpg',
                             'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/mpg_scale']},
            'pyrim': {'loader': _load_train_scale,
                      'args': ['https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/pyrim',
                               'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/pyrim_scale']},
            'space_ga': {'loader': _load_train_scale,
                         'args': ['https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/space_ga',
                                  'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/space_ga_scale']},
            'triazines': {'loader': _load_train_scale,
                          'args': ['https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/triazines',
                                   'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/triazines_scale']},
            'E2006-log1p': {'loader': _load_train_scale,
                            'args': ['https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/log1p.E2006.train.bz2',
                                     'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/log1p.E2006.test.bz2',
                                     fetch_bz2]},
            'E2006-tfidf': {'loader': _load_train_scale,
                            'args': ['https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/E2006.train.bz2',
                                     'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/E2006.test.bz2',
                                     fetch_bz2]},
            'YearPredictionMSD': {'loader': _load_train_scale,
                                  'args': ['https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/YearPredictionMSD.bz2',
                                           'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/YearPredictionMSD.t.bz2',
                                           fetch_bz2]}}


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
    X, y, X_test, y_test, inner_cv = datasets[name]['loader'](name, *datasets[name]['args'])
    if return_X_y:
        return X, y, X_test, y_test, inner_cv, None
    return Bunch(data=X, target=y, data_test=X_test, target_test=y_test,
                 inner_cv=inner_cv, DESCR=name)
