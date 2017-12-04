"""
LIBSVM cod-rna dataset.

@author: David Diaz Vico
@license: MIT
"""

from ..base import load_train_test_remaining


def load_cod_rna(return_X_y=False):
    """Load cod-rna dataset.

    Loads the cod-rna dataset.

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
    return load_train_test_remaining('cod-rna',
                                     'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/cod-rna',
                                     'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/cod-rna.t',
                                     'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/cod-rna.r',
                                     return_X_y=return_X_y)
