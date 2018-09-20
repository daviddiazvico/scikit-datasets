"""
Gunnar Raetsch benchmark datasets
(https://github.com/tdiethe/gunnar_raetsch_benchmark_datasets).

@author: David Diaz Vico
@license: MIT
"""

from scipy.io import loadmat
from sklearn.datasets.base import Bunch

import numpy as np

from .base import CustomSplit, fetch_file

datasets = {'banana': {}, 'breast_cancer': {}, 'diabetis': {},
            'flare_solar': {}, 'german': {}, 'heart': {}, 'image': {},
            'ringnorm': {}, 'splice': {}, 'thyroid': {}, 'titanic': {},
            'twonorm': {}, 'waveform': {}}


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
    filename = fetch_file('gunnar_raetsch',
                          'https://github.com/tdiethe/gunnar_raetsch_benchmark'
                          '_datasets/raw/master/benchmarks.mat')
    X, y, train_splits, test_splits = loadmat(filename)[name][0][0]
    if (len(y.shape) == 1) or (np.prod(y.shape[1:]) == 1):
        y = y.ravel()
    y[y == -1] = 0
    inner_cv = CustomSplit(train_splits[:5] - 1, test_splits[:5] - 1)
    outer_cv = CustomSplit(train_splits - 1, test_splits - 1)
    if return_X_y:
        return X, y, None, None, inner_cv, outer_cv
    return Bunch(data=X, target=y, inner_cv=inner_cv, outer_cv=outer_cv,
                 DESCR=name)
