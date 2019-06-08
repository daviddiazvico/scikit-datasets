"""
@author: David Diaz Vico
@license: MIT
"""

from sklearn.model_selection import GridSearchCV

from skdatasets.utils.estimator import json2estimator


def test_json2estimator():
    """Tests instantiation of estimator from a json file."""
    import sklearn
    e = json2estimator('tests/utils/GridSearchCVLinearRegression.json')
    assert type(e) == GridSearchCV


def test_json2estimator_custom():
    """Tests instantiation of a custom estimator from a json file."""
    import skdatasets
    e = json2estimator('tests/utils/GridSearchCVLinearRegressionCustom.json')
    assert type(e) == GridSearchCV
