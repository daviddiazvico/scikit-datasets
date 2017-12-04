"""
Keel glass5 dataset.

@author: David Diaz Vico
@license: MIT
"""

from ..base import load_imbalanced


def load_glass5(return_X_y=False):
    """Load glass5 dataset.

    Loads the glass5 dataset.

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
    return load_imbalanced('glass5',
                           'http://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRhigherThan9p1',
                           names=['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'Class'],
                           target_names=['Class'], return_X_y=return_X_y)
