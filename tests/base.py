"""
Tests.

@author: David Diaz Vico
@license: MIT
"""

import numpy as np


def check_array(a, n_rows, n_cols=None):
    """Checks if an array has the correct number of rows and cols."""
    if n_cols is not None:
        assert a.shape == (n_rows, n_cols)
    else:
        assert len(a) == n_rows


def check_folds(folds, n_patterns, n_folds):
    """Checks if some folds are correct."""
    assert len(folds) == n_patterns
    assert 1 <= len(np.unique(folds)) <= n_folds + 1


def check_load_dataset(load, n_patterns, n_variables, array_names,
                       n_targets=None, n_folds=None):
    """Checks that a dataset is loaded correctly."""

    try:
        (X, y), (X_tr, y_tr), (X_val, y_val), (X_test, y_test) = load(return_X_y=True)
        partitions = ((X, y), (X_tr, y_tr), (X_val, y_val), (X_test, y_test))
    except:
        try:
            (X, y), (X_test, y_test) = load(return_X_y=True)
            partitions = ((X, y), (X_test, y_test))
        except:
            X, y = load(return_X_y=True)
            partitions = ((X, y), )
    for (X, y), n_pattern in zip(partitions, n_patterns):
        check_array(X, n_pattern, n_cols=n_variables)
        check_array(y, n_pattern, n_cols=n_targets)
    bunch = load()
    for (X_name, y_name), n_pattern in zip(array_names, n_patterns):
        check_array(bunch[X_name], n_pattern, n_cols=n_variables)
        check_array(bunch[y_name], n_pattern, n_cols=n_targets)
    if n_folds is not None:
        for n_fold in n_folds:
            check_folds(bunch['fold' + str(n_fold)], n_patterns[0],
                        n_folds=n_fold)
