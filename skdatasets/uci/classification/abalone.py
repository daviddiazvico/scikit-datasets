"""
UCI abalone dataset.

@author: David Diaz Vico
@license: MIT
"""

from ..base import load_train


def load_abalone(return_X_y=False):
    """Load abalone dataset.

    Loads the abalone dataset.

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
    return load_train('abalone',
                      'http://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data',
                      'http://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.names',
                      ['Sex', 'Length', 'Diameter', 'Height', 'Whole weight',
                       'Shucked weight', 'Viscera weight', 'Shell weight',
                       'Rings'], ['Rings'], return_X_y=return_X_y)
