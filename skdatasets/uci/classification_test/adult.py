"""
UCI adult dataset.

@author: David Diaz Vico
@license: MIT
"""

from ..base import load_train_test


def load_adult(return_X_y=False):
    """Load adult dataset.

    Loads the adult dataset.

    Parameters
    ----------
    return_X_y: bool, default=False
                If True, returns (data, target) instead of a Bunch object..

    Returns
    -------
    data: Bunch
          Dictionary-like object with all the data and metadata.
    (X, y), (X_test, y_test): lists of arrays
                              If return_X_y is True

    """
    return load_train_test('adult',
                           'http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data',
                           'http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test',
                           'http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names',
                           ['age', 'workclass', 'fnlwgt', 'education',
                            'education-num', 'marital-status', 'occupation',
                            'relationship', 'race', 'sex', 'capital-gain',
                            'capital-loss', 'hours-per-week', 'native-country',
                            'target'], ['target'], return_X_y=return_X_y)
