"""
LIBSVM a4a dataset.

@author: David Diaz Vico
@license: MIT
"""

from ..base import load_train_test


def load_a4a(return_X_y=False):
    """Load a4a dataset.

    Loads the a4a dataset.

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
    return load_train_test('a4a',
                           'http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a4a',
                           'http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a4a.t',
                           return_X_y=return_X_y)
