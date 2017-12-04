"""
LIBSVM a8a dataset.

@author: David Diaz Vico
@license: MIT
"""

from ..base import load_train_test


def load_a8a(return_X_y=False):
    """Load a8a dataset.

    Loads the a8a dataset.

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
    return load_train_test('a8a',
                           'http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a8a',
                           'http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a8a.t',
                           return_X_y=return_X_y)
