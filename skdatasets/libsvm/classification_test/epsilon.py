"""
LIBSVM epsilon dataset.

@author: David Diaz Vico
@license: MIT
"""

from ..base import load_train_test


def load_epsilon(return_X_y=False):
    """Load epsilon dataset.

    Loads the epsilon dataset.

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
    return load_train_test('epsilon',
                           'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/epsilon_normalized.bz2',
                           'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/epsilon_normalized.t.bz2',
                           return_X_y=return_X_y)
