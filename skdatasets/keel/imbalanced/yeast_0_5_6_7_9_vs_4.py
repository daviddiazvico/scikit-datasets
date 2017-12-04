"""
Keel yeast-0-5-6-7-9_vs_4 dataset.

@author: David Diaz Vico
@license: MIT
"""

from ..base import load_imbalanced


def load_yeast_0_5_6_7_9_vs_4(return_X_y=False):
    """Load yeast-0-5-6-7-9_vs_4 dataset.

    Loads the yeast-0-5-6-7-9_vs_4 dataset.

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
    return load_imbalanced('yeast-0-5-6-7-9_vs_4',
                           'http://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRhigherThan9p1',
                           names=['Mcg', 'Gvh', 'Alm', 'Mit', 'Erl', 'Pox',
                                  'Vac', 'Nuc', 'Class'],
                           target_names=['Class'], return_X_y=return_X_y)
