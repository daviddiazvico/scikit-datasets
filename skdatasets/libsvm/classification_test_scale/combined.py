"""
LIBSVM combined dataset.

@author: David Diaz Vico
@license: MIT
"""

from ..base import load_train_test_scale
from ...base import fetch_bz2


def load_combined(return_X_y=False):
    """Load combined dataset.

    Loads the combined dataset.

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
    return load_train_test_scale('combined',
                                 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/vehicle/combined.bz2',
                                 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/vehicle/combined.t.bz2',
                                 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/vehicle/combined_scale.bz2',
                                 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/vehicle/combined_scale.t.bz2',
                                 fetch_file=fetch_bz2, return_X_y=return_X_y)
