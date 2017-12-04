"""
Keel new-thyroid dataset.

@author: David Diaz Vico
@license: MIT
"""

from ..base import load_standard_classification


def load_newthyroid(return_X_y=False):
    """Load newthyroid dataset.

    Loads the newthyroid dataset.

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
    return load_standard_classification('newthyroid',
                                        'http://sci2s.ugr.es/keel/dataset/data/classification/',
                                        names=['T3resin', 'thyroxin',
                                               'triiodothyronine',
                                               'thyroidstimulating',
                                               'TSH_value', 'class'],
                                        target_names=['class'],
                                        return_X_y=return_X_y)
