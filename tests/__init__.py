"""
Tests.

@author: David Diaz Vico
@license: MIT
"""

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def check_estimator(data):
    """Check that the dataset can be used to cross-validate an estimator."""
    estimator = GridSearchCV(Pipeline([('tr', StandardScaler(with_mean=False)),
                                       ('pred', Ridge(max_iter=4))]),
                             {'pred__alpha': [0.33, 0.66]},
                             cv=data.inner_cv, error_score=np.nan)
    if data.data_test is not None:
        estimator.fit(data.data, y=data.target)
        estimator.score(data.data_test, y=data.target_test)
    else:
        if hasattr(data.outer_cv, '__iter__'):
            for X, y, X_test, y_test in data.outer_cv:
                estimator.fit(X, y=y)
                estimator.score(X_test, y=y_test)
        else:
            cross_validate(estimator, data.data, y=data.target,
                           cv=data.outer_cv)
