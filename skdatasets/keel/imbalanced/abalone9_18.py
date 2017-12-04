"""
Keel abalone9-18 dataset.

@author: David Diaz Vico
@license: MIT
"""

from ..base import load_imbalanced


def load_abalone9_18(return_X_y=False):
    """Load abalone9-18 dataset.

    Loads the abalone9-18 dataset.

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
    return load_imbalanced('abalone9-18',
                           'http://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRhigherThan9p1',
                           names=['Sex', 'Length', 'Diameter', 'Height',
                                  'Whole_weight', 'Shucked_weight',
                                  'Viscera_weight', 'Shell_weight', 'Class'],
                           target_names=['Class'], return_X_y=return_X_y)
