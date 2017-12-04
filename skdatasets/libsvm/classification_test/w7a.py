"""
LIBSVM w7a dataset.

@author: David Diaz Vico
@license: MIT
"""

from ..base import load_train_test


def load_w7a(return_X_y=False):
    """Load w7a dataset.

    Loads the w7a dataset.

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
    return load_train_test('w7a',
                           'http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/w7a',
                           'http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/w7a.t',
                           return_X_y=return_X_y)
