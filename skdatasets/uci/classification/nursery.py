"""
UCI nursery dataset.

@author: David Diaz Vico
@license: MIT
"""

from ..base import load_train


def load_nursery(return_X_y=False):
    """Load nursery dataset.

    Loads the nursery dataset.

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
    return load_train('nursery',
                      'https://archive.ics.uci.edu/ml/machine-learning-databases/nursery/nursery.data',
                      'https://archive.ics.uci.edu/ml/machine-learning-databases/nursery/nursery.names',
                      ['parents', 'has_nurs', 'form', 'children', 'housing',
                       'finance', 'social', 'health', 'target'],
                      ['target'], return_X_y=return_X_y)
