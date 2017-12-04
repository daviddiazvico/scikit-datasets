"""
Keel glass dataset.

@author: David Diaz Vico
@license: MIT
"""

from ..base import load_standard_classification


def load_glass(return_X_y=False):
    """Load glass dataset.

    Loads the glass dataset.

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
    return load_standard_classification('glass',
                                        'http://sci2s.ugr.es/keel/dataset/data/classification',
                                        names=['RI', 'Na', 'Mg', 'Al', 'Si',
                                               'K', 'Ca', 'Ba', 'Fe',
                                               'typeGlass'],
                                        target_names=['typeGlass'],
                                        return_X_y=return_X_y)
