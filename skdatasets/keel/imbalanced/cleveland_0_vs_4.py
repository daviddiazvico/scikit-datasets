"""
Keel cleveland-0_vs_4 dataset.

@author: David Diaz Vico
@license: MIT
"""

from ..base import load_imbalanced


def load_cleveland_0_vs_4(return_X_y=False):
    """Load cleveland-0_vs_4 dataset.

    Loads the cleveland-0_vs_4 dataset.

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
    return load_imbalanced('cleveland-0_vs_4',
                           'http://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRhigherThan9p2',
                           names=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
                                  'restecg', 'thalach', 'exang', 'oldpeak',
                                  'slope', 'ca', 'thal', 'num'],
                           target_names=['num'], return_X_y=return_X_y)
