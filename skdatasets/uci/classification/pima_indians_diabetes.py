"""
UCI pima-indians-diabetes dataset.

@author: David Diaz Vico
@license: MIT
"""

from ..base import load_train


def load_pima_indians_diabetes(return_X_y=False):
    """Load pima-indians-diabetes dataset.

    Loads the pima-indians-diabetes dataset.

    Parameters
    ----------
    return_X_y: bool, default=False
                If True, returns (data, target) instead of a Bunch object..

    Returns
    -------
    data: Bunch
          Dictionary-like object with all the data and metadata.
    X, y: arrays
          If return_X_y is True

    """
    return load_train('pima-indians-diabetes',
                      'https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data',
                      'https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.names',
                      ['1', '2', '3', '4', '5', '6', '7', '8', 'target'],
                      ['target'], return_X_y=return_X_y)
