"""
Keel glass-0-1-4-6_vs_2 dataset.

@author: David Diaz Vico
@license: MIT
"""

from ..base import load_imbalanced


def load_glass_0_1_4_6_vs_2(return_X_y=False):
    """Load glass-0-1-4-6_vs_2 dataset.

    Loads the glass-0-1-4-6_vs_2 dataset.

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
    return load_imbalanced('glass-0-1-4-6_vs_2',
                           'http://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRhigherThan9p2',
                           names=['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba',
                                  'Fe', 'typeGlass'],
                           target_names=['typeGlass'], return_X_y=return_X_y)
