"""
LIBSVM mpg dataset.

@author: David Diaz Vico
@license: MIT
"""

from ..base import load_train_scale


def load_mpg(return_X_y=False):
    """Load mpg dataset.

    Loads the mpg dataset.

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
    return load_train_scale('mpg',
                            'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/mpg',
                            'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/mpg_scale',
                            return_X_y=return_X_y)
