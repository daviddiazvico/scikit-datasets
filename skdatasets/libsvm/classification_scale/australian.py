"""
LIBSVM australian dataset.

@author: David Diaz Vico
@license: MIT
"""

from ...libsvm.base import load_train_scale


def load_australian(return_X_y=False):
    """Load australian dataset.

    Loads the australian dataset.

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
    return load_train_scale('australian',
                            'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/australian',
                            'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/australian_scale',
                            return_X_y=return_X_y)
