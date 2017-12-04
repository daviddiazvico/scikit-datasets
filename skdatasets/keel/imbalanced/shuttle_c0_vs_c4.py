"""
Keel shuttle-c0-vs-c4 dataset.

@author: David Diaz Vico
@license: MIT
"""


from ..base import load_imbalanced


def load_shuttle_c0_vs_c4(return_X_y=False):
    """Load shuttle-c0-vs-c4 dataset.

    Loads the shuttle-c0-vs-c4 dataset.

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
    return load_imbalanced('shuttle-c0-vs-c4',
                           'http://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRhigherThan9p1',
                           names=['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7',
                                  'A8', 'A9', 'Class'],
                           target_names=['Class'], return_X_y=return_X_y)
