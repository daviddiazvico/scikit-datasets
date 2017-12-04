"""
LIBSVM pendigits dataset.

@author: David Diaz Vico
@license: MIT
"""

from ..base import load_train_test


def load_pendigits(return_X_y=False):
    """Load pendigits dataset.

    Loads the pendigits dataset.

    Parameters
    ----------
    return_X_y: bool, default=False
                If True, returns (data, target) instead of a Bunch object..

    Returns
    -------
    data: Bunch
          Dictionary-like object with all the data and metadata.
    (X, y), (X_test, y_test): lists of arrays
                              If return_X_y is True

    """
    return load_train_test('pendigits',
                           'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/pendigits',
                           'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/pendigits.t',
                           return_X_y=return_X_y)
