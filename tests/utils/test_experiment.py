"""
@author: David Diaz Vico
@license: MIT
"""

from sacred.observers import FileStorageObserver
from sklearn.datasets import load_boston
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeRegressor

from skdatasets.utils.experiment import experiment


def _dataset(inner_cv=None, outer_cv=None):
    data = load_boston()
    if outer_cv is None:
        X, X_test, y, y_test = train_test_split(data.data, data.target)
        data.data = X
        data.target = y
        data.data_test = X_test
        data.target_test = y_test
        data.outer_cv = None
    else:
        data.data_test = data.target_test = None
        data.outer_cv = outer_cv
    data.inner_cv = inner_cv
    return data


def _estimator(cv):
    return GridSearchCV(DecisionTreeRegressor(), {'max_depth': [2, 4]},
                        iid=True, cv=cv)


def _experiment(inner_cv, outer_cv):
    e = experiment(_dataset, _estimator)
    e.observers.append(FileStorageObserver.create('.results'))
    e.run(config_updates={'dataset': {'inner_cv': inner_cv,
                                      'outer_cv': outer_cv}})


def test_nested_cv():
    """Tests nested CV experiment."""
    _experiment(3, 3)


def test_inner_cv():
    """Tests inner CV experiment."""
    _experiment(3, None)


def test_explicit_inner_folds():
    """Tests explicit inner folds experiment."""
    X, y = load_boston(return_X_y=True)
    _experiment([[X[:10], y[:10], X[10:20], y[10:20]],
                 [X[10:20], y[10:20], X[20:30], y[20:30]],
                 [X[20:30], y[20:30], X[30:40], y[30:40]]], 3)


def test_explicit_outer_folds():
    """Tests explicit outer folds experiment."""
    X, y = load_boston(return_X_y=True)
    _experiment(3, [[X[:10], y[:10], X[10:20], y[10:20]],
                    [X[10:20], y[10:20], X[20:30], y[20:30]],
                    [X[20:30], y[20:30], X[30:40], y[30:40]]])


def test_explicit_nested_folds():
    """Tests explicit nested folds experiment."""
    X, y = load_boston(return_X_y=True)
    _experiment([[X[:10], y[:10], X[10:20], y[10:20]],
                 [X[10:20], y[10:20], X[20:30], y[20:30]],
                 [X[20:30], y[20:30], X[30:40], y[30:40]]],
                [[X[:10], y[:10], X[10:20], y[10:20]],
                 [X[10:20], y[10:20], X[20:30], y[20:30]],
                 [X[20:30], y[20:30], X[30:40], y[30:40]]])
