"""
LIBSVM cadata dataset.

@author: David Diaz Vico
@license: MIT
"""

from ..base import load_train


def load_cadata(return_X_y=False):
    """Load cadata dataset.

    Loads the cadata dataset.

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
    return load_train('cadata',
                      'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/cadata',
                      return_X_y=return_X_y)
