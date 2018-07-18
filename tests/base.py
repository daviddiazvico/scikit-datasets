"""
Tests.

@author: David Diaz Vico
@license: MIT
"""


def check_array(a, n_rows, n_cols=None):
    """Checks if an array has the correct number of rows and cols."""
    if n_cols is not None:
        assert a.shape == (n_rows, n_cols)
    else:
        assert len(a) == n_rows


def check_folds(bunch, n_patterns, n_folds):
    """Checks if some folds are correct."""
    X = bunch['data' + str(n_folds)]
    y = bunch['target' + str(n_folds)]
    X_test = bunch['data' + str(n_folds) + '_test']
    y_test = bunch['target' + str(n_folds) + '_test']
    assert len(X) == len(y) == len(X_test) == len(y_test) == n_folds
    for X_fold, y_fold, X_test_fold, y_test_fold in zip(X, y, X_test, y_test):
        assert len(X_fold) == len(y_fold)
        assert len(X_fold) < n_patterns
        assert len(y_fold) < n_patterns
        assert len(X_test_fold) == len(y_test_fold)
        assert len(X_test_fold) < n_patterns
        assert len(y_test_fold) < n_patterns


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
            check_folds(bunch, n_patterns[0], n_folds=n_fold)


def check_items(not_nones, nones):
    """Checks if items are None or not."""
    for item in not_nones:
        assert item is not None
    for item in nones:
        assert item is None
