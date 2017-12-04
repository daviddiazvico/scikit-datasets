"""
Keel balance dataset.

@author: David Diaz Vico
@license: MIT
"""

from ..base import load_standard_classification


def load_balance(return_X_y=False):
    """Load balance dataset.

    Loads the balance dataset.

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
    return load_standard_classification('balance',
                                        'http://sci2s.ugr.es/keel/dataset/data/classification',
                                        names=['left-weight', 'left-distance',
                                               'right-weight', 'right-distance',
                                               'class'], target_names=['class'],
                                        return_X_y=return_X_y)
