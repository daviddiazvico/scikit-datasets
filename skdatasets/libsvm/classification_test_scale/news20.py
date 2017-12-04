"""
LIBSVM news20 dataset.

@author: David Diaz Vico
@license: MIT
"""

from ..base import load_train_test_scale
from ...base import fetch_bz2


def load_news20(return_X_y=False):
    """Load news20 dataset.

    Loads the news20 dataset.

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
    return load_train_test_scale('news20',
                                 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/news20.bz2',
                                 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/news20.t.bz2',
                                 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/news20.scale.bz2',
                                 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/news20.t.scale.bz2',
                                 fetch_file=fetch_bz2, return_X_y=return_X_y)
