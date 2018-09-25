"""
Tests.

@author: David Diaz Vico
@license: MIT
"""

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (BaseCrossValidator, cross_val_score,
                                     GridSearchCV)

import numpy as np


def load(loader):
    """Checks that a dataset is loaded correctly."""
    X, y, X_test, y_test, inner_cv, outer_cv = loader(return_X_y=True)
    bunch = loader()
    assert isinstance(X, np.ndarray)
    assert isinstance(bunch.data, np.ndarray)
    assert X.shape == bunch.data.shape
    if y is not None:
        assert isinstance(y, np.ndarray)
        assert isinstance(bunch.target, np.ndarray)
        assert (X.shape[0] == y.shape[0] ==
                bunch.data.shape[0] == bunch.target.shape[0])
    if X_test is not None:
        assert isinstance(X_test, np.ndarray)
        assert isinstance(bunch.data_test, np.ndarray)
        assert X_test.shape[0] == bunch.data_test.shape[0]
    if y_test is not None:
        assert isinstance(y_test, np.ndarray)
        assert isinstance(bunch.target_test, np.ndarray)
        assert (X_test.shape[0] == y_test.shape[0] ==
                bunch.data_test.shape[0] == bunch.target_test.shape[0])
    if inner_cv is not None:
        assert isinstance(inner_cv, BaseCrossValidator)
        assert isinstance(bunch.inner_cv, BaseCrossValidator)
    if outer_cv is not None:
        assert isinstance(outer_cv, BaseCrossValidator)
        assert isinstance(bunch.outer_cv, BaseCrossValidator)


def use(loader):
    """Checks that a dataset can be used correctly."""
    X, y, X_test, y_test, inner_cv, outer_cv = loader(return_X_y=True)
    param_grid = {'penalty': ['l1', 'l2']}
    estimator = GridSearchCV(LogisticRegression(), param_grid=param_grid,
                             cv=inner_cv)
    if (X_test is not None) and (y_test is not None):
        # Test split for scoring
        estimator.fit(X, y)
        estimator.score(X_test, y_test)
    else:
        # CV scoring
        if isinstance(inner_cv,
                      BaseCrossValidator) or hasattr(inner_cv, '__iter__'):
            # Validation split for hyperparameter tuning
            estimator.fit(X, y)
            estimator = estimator.best_estimator_
        cross_val_score(estimator, X, y, cv=outer_cv)
