"""
LIBSVM bodyfat dataset.

@author: David Diaz Vico
@license: MIT
"""

from ..base import load_train_scale


def load_bodyfat(return_X_y=False):
    """Load bodyfat dataset.

    Loads the bodyfat dataset.

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
    return load_train_scale('bodyfat',
                            'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/bodyfat',
                            'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/bodyfat_scale',
                            return_X_y=return_X_y)
