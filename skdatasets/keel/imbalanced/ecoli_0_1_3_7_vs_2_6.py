"""
Keel ecoli-0-1-3-7_vs_2-6 dataset.

@author: David Diaz Vico
@license: MIT
"""

from ..base import load_imbalanced


def load_ecoli_0_1_3_7_vs_2_6(return_X_y=False):
    """Load ecoli-0-1-3-7_vs_2-6 dataset.

    Loads the ecoli-0-1-3-7_vs_2-6 dataset.

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
    return load_imbalanced('ecoli-0-1-3-7_vs_2-6',
                           'http://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRhigherThan9p1',
                           names=['Mcg', 'Gvh', 'Lip', 'Chg', 'Aac', 'Alm1',
                                  'Alm2', 'Class'],
                           target_names=['Class'], return_X_y=return_X_y)
