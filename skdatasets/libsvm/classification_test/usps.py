"""
LIBSVM usps dataset.

@author: David Diaz Vico
@license: MIT
"""

from ..base import load_train_test
from ...base import fetch_bz2


def load_usps(return_X_y=False):
    """Load usps dataset.

    Loads the usps dataset.

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
    return load_train_test('usps',
                           'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/usps.bz2',
                           'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/usps.t.bz2',
                           fetch_file=fetch_bz2, return_X_y=return_X_y)
