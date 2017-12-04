"""
LIBSVM triazines dataset.

@author: David Diaz Vico
@license: MIT
"""

from ..base import load_train_scale


def load_triazines(return_X_y=False):
    """Load triazines dataset.

    Loads the triazines dataset.

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
    return load_train_scale('triazines',
                            'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/triazines',
                            'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/triazines_scale',
                            return_X_y=return_X_y)
