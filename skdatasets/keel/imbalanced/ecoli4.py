"""
Keel ecoli4 dataset.

@author: David Diaz Vico
@license: MIT
"""

from ..base import load_imbalanced


def load_ecoli4(return_X_y=False):
    """Load ecoli4 dataset.

    Loads the ecoli4 dataset.

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
    return load_imbalanced('ecoli4',
                           'http://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRhigherThan9p1',
                           names=['Mcg', 'Gvh', 'Lip', 'Chg', 'Aac', 'Alm1',
                                  'Alm2', 'Class'],
                           target_names=['Class'], return_X_y=return_X_y)
