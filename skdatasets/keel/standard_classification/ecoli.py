"""
Keel ecoli dataset.

@author: David Diaz Vico
@license: MIT
"""

from ..base import load_standard_classification


def load_ecoli(return_X_y=False):
    """Load ecoli dataset.

    Loads the ecoli dataset.

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
    return load_standard_classification('ecoli',
                                        'http://sci2s.ugr.es/keel/dataset/data/classification',
                                        names=['mcg', 'gvh', 'lip', 'chg',
                                               'aac', 'alm1', 'alm2', 'class'],
                                        target_names=['class'],
                                        return_X_y=return_X_y)
