"""
Tests.

@author: David Diaz Vico
@license: MIT
"""

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def check_estimator(data):
    """Check that the dataset can be used to cross-validate an estimator."""
    estimator = GridSearchCV(Pipeline([('tr', StandardScaler(with_mean=False)),
                                       ('pred', Ridge(max_iter=4))]),
                             {'pred__alpha': [0.33, 0.66]},
                             cv=data.inner_cv, error_score=np.nan)
    if data.train_indexes is not None and data.test_indexes is not None:

        train_indexes = data.train_indexes

        train_indexes += data.validation_indexes

        estimator.fit(
            data.data[train_indexes],
            y=data.target[train_indexes],
        )
        estimator.score(
            data.data[data.test_indexes],
            y=data.target[data.test_indexes]
        )
    else:
        if hasattr(data.outer_cv, '__iter__'):
            for X, y, X_test, y_test in data.outer_cv:
                estimator.fit(X, y=y)
                estimator.score(X_test, y=y_test)
        else:
            cross_validate(estimator, data.data, y=data.target,
                           cv=data.outer_cv)
