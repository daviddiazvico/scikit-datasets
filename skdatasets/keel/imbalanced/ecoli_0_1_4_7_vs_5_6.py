"""
Keel ecoli-0-1-4-7_vs_5-6 dataset.

@author: David Diaz Vico
@license: MIT
"""

from ..base import load_imbalanced


def load_ecoli_0_1_4_7_vs_5_6(return_X_y=False):
    """Load ecoli-0-1-4-7_vs_5-6 dataset.

    Loads the ecoli-0-1-4-7_vs_5-6 dataset.

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
    return load_imbalanced('ecoli-0-1-4-7_vs_5-6',
                           'http://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRhigherThan9p2',
                           names=['a1', 'a2', 'a3', 'a5', 'a6', 'a7', 'class'],
                           target_names=['class'], return_X_y=return_X_y)
